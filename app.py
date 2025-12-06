import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import fitz  # PyMuPDF
from PyQt6 import QtCore, QtGui, QtWidgets
from PIL import Image
from dotenv import load_dotenv

# google-generativeai가 Python 3.10 미만에서 importlib.metadata.packages_distributions
# 를 필요로 하기 때문에 백포트 모듈이 있으면 주입한다.
try:  # pragma: no cover - shim setup
    import importlib.metadata as _stdlib_metadata
except ImportError:  # pragma: no cover - Python<3.8 fallback
    import importlib_metadata as _stdlib_metadata  # type: ignore

try:  # pragma: no cover - optional dependency
    import importlib_metadata as _backport_metadata  # type: ignore
except ImportError:  # pragma: no cover - backport not installed
    _backport_metadata = None

if not hasattr(_stdlib_metadata, "packages_distributions") and _backport_metadata:
    if hasattr(_backport_metadata, "packages_distributions"):
        _stdlib_metadata.packages_distributions = _backport_metadata.packages_distributions  # type: ignore[attr-defined]

import google.generativeai as genai

# Load variables like GOOGLE_API_KEY from a local .env file when present.
load_dotenv()


@dataclass
class Chapter:
    """Container that represents a single PDF page (or contiguous range)."""

    id: str
    title: str
    start_page: int  # zero-based inclusive
    end_page: int  # zero-based inclusive
    summary: Optional[str] = None

    def contains(self, page_index: int) -> bool:
        return self.start_page <= page_index <= self.end_page


PROMPT_TEMPLATE = """당신은 PDF 학습 자료를 이해하기 쉽게 설명하는 도우미입니다.
주어진 페이지 스캔 이미지와 텍스트, 직전/다음 페이지의 문맥을 함께 참고하여 학습 흐름이 이어지도록 설명하세요.
- 친절하고 명확한 한국어로 3~5개의 bullet point로 요약합니다.
- 각 bullet point는 한 문단이 아니라 1~2문장으로 요약합니다.
- 텍스트뿐 아니라 페이지에 포함된 시각 자료(도표/이미지)의 의미를 자연스럽게 설명에 녹여 주세요.
- 마지막에는 '복습 질문'이라는 소제목 아래에 페이지 전체를 확인할 수 있는 2개의 질문을 제시합니다.

페이지 정보: "{title}"

[이전 페이지 문맥]
{previous}

[현재 페이지 내용]
{current}

[다음 페이지 문맥]
{next}
"""


def hash_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def build_chapters(doc: fitz.Document) -> List[Chapter]:
    chapters: List[Chapter] = []
    for page_index in range(doc.page_count):
        chapters.append(
            Chapter(
                id=f"page_{page_index:04d}",
                title=f"페이지 {page_index + 1}",
                start_page=page_index,
                end_page=page_index,
            )
        )
    return chapters


def build_prompt(chapter: Chapter, current: str, previous: str, next_text: str) -> str:
    prev_section = previous or "이전 페이지 텍스트가 없거나 필요하지 않습니다."
    next_section = next_text or "다음 페이지 텍스트가 없거나 필요하지 않습니다."
    return PROMPT_TEMPLATE.format(
        title=chapter.title,
        previous=prev_section,
        current=current,
        next=next_section,
    )


def neighbor_chapters(chapters: List[Chapter], target: Chapter) -> tuple[Optional[Chapter], Optional[Chapter]]:
    for idx, chapter in enumerate(chapters):
        if chapter.id == target.id:
            prev_chapter = chapters[idx - 1] if idx > 0 else None
            next_chapter = chapters[idx + 1] if idx + 1 < len(chapters) else None
            return prev_chapter, next_chapter
    return None, None


def extract_text_for_range(
    pdf_path: Path, start_page: int, end_page: int, max_chars: int
) -> str:
    buffer: List[str] = []
    total_len = 0
    with fitz.open(pdf_path) as doc:
        for page_index in range(start_page, end_page + 1):
            if page_index < 0 or page_index >= doc.page_count:
                continue
            page = doc.load_page(page_index)
            text = page.get_text("text")
            if not text:
                continue
            remaining = max_chars - total_len
            if remaining <= 0:
                break
            to_take = text[:remaining]
            buffer.append(to_take)
            total_len += len(to_take)
            if total_len >= max_chars:
                break
    return "\n".join(buffer)


def extract_contextual_texts(
    pdf_path: Path,
    current: Chapter,
    previous: Optional[Chapter],
    next_chapter: Optional[Chapter],
    max_chars: int,
    context_chars: int,
) -> tuple[str, str, str]:
    current_text = extract_text_for_range(
        pdf_path,
        current.start_page,
        current.end_page,
        max_chars,
    )
    prev_text = (
        extract_text_for_range(pdf_path, previous.start_page, previous.end_page, context_chars)
        if previous
        else ""
    )
    next_text = (
        extract_text_for_range(pdf_path, next_chapter.start_page, next_chapter.end_page, context_chars)
        if next_chapter
        else ""
    )
    return current_text, prev_text, next_text


FALLBACK_TEXT = "이 페이지에서 텍스트를 추출하지 못했습니다. 이미지나 도표를 중심으로 설명해 주세요."


def _resize_image(image: Image.Image, max_pixels: int) -> Image.Image:
    aspect = image.width / image.height if image.height else 1
    max_side = int((max_pixels * aspect) ** 0.5)
    if max_side < 1:
        max_side = 1
    max_width = min(image.width, max_side)
    max_height = max(1, int(max_width / aspect))
    resized = image.copy()
    resized.thumbnail((max_width, max_height), Image.LANCZOS)
    return resized


_shared_qt_app: Optional[QtWidgets.QApplication] = None


def ensure_qapp() -> QtWidgets.QApplication:
    global _shared_qt_app
    if _shared_qt_app is not None:
        return _shared_qt_app
    instance = QtWidgets.QApplication.instance()
    if instance is None:
        instance = QtWidgets.QApplication([])
    _shared_qt_app = instance
    return instance


def render_page_image(pdf_path: Path, page_index: int, max_pixels: int) -> Optional[Image.Image]:
    if max_pixels <= 0:
        return None
    with fitz.open(pdf_path) as doc:
        if page_index < 0 or page_index >= doc.page_count:
            return None
        page = doc.load_page(page_index)
        pix = page.get_pixmap(alpha=False)
        image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        if image.width * image.height > max_pixels:
            image = _resize_image(image, max_pixels)
        return image


class GeminiExplainer:
    def __init__(
        self,
        api_key: str,
        model_name: str,
        pdf_hash: str,
        cache_path: Path,
    ) -> None:
        if not api_key:
            raise ValueError(
                "Gemini API 키가 필요합니다. --api-key 옵션 또는 GOOGLE_API_KEY 환경 변수를 확인하세요."
            )
        self.model_name = model_name
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        self.pdf_hash = pdf_hash
        self.cache_path = cache_path
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache = self._load_cache()

    def _load_cache(self) -> dict:
        if self.cache_path.exists():
            try:
                return json.loads(self.cache_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                return {}
        return {}

    def _save_cache(self) -> None:
        self.cache_path.write_text(json.dumps(self.cache, ensure_ascii=False, indent=2))

    def _cache_key(self, chapter_id: str) -> str:
        return f"{self.pdf_hash}:{chapter_id}"

    def get_cached(self, chapter: Chapter) -> Optional[str]:
        return self.cache.get(self._cache_key(chapter.id))

    def store(self, chapter: Chapter, summary: str) -> None:
        self.cache[self._cache_key(chapter.id)] = summary
        self._save_cache()

    def explain(
        self,
        chapter: Chapter,
        chapter_text: str,
        previous_text: str = "",
        next_text: str = "",
        images: Optional[List[Image.Image]] = None,
    ) -> str:
        prompt = build_prompt(chapter, chapter_text, previous_text, next_text)
        parts: List[object] = [prompt]
        if images:
            parts.extend(images)
        response = self.model.generate_content(parts)
        output = getattr(response, "text", None)
        if not output and hasattr(response, "candidates"):
            output = "\n".join(
                part.text for cand in response.candidates for part in cand.content.parts if getattr(part, "text", None)
            )
        if not output:
            raise RuntimeError("Gemini 응답에 텍스트가 포함되어 있지 않습니다.")
        cleaned = output.strip()
        self.store(chapter, cleaned)
        return cleaned


class WorkerSignals(QtCore.QObject):
    completed = QtCore.pyqtSignal(str, str)
    failed = QtCore.pyqtSignal(str, str)


class SummaryWorker(QtCore.QRunnable):
    def __init__(
        self,
        pdf_path: Path,
        chapter: Chapter,
        chapters: List[Chapter],
        explainer: GeminiExplainer,
        max_chars: int,
        context_chars: int,
        max_image_pixels: int,
    ) -> None:
        super().__init__()
        self.pdf_path = pdf_path
        self.chapter = chapter
        self.chapters = chapters
        self.explainer = explainer
        self.max_chars = max_chars
        self.context_chars = context_chars
        self.max_image_pixels = max_image_pixels
        self.signals = WorkerSignals()

    @QtCore.pyqtSlot()
    def run(self) -> None:
        try:
            prev_chapter, next_chapter = neighbor_chapters(self.chapters, self.chapter)
            current_text, prev_text, next_text = extract_contextual_texts(
                self.pdf_path,
                self.chapter,
                prev_chapter,
                next_chapter,
                self.max_chars,
                self.context_chars,
            )
            if not current_text:
                current_text = FALLBACK_TEXT
            image = render_page_image(
                self.pdf_path,
                self.chapter.start_page,
                self.max_image_pixels,
            )
            images = [image] if image else None
            summary = self.explainer.explain(
                self.chapter,
                current_text,
                previous_text=prev_text,
                next_text=next_text,
                images=images,
            )
            self.signals.completed.emit(self.chapter.id, summary)
        except Exception as exc:  # pragma: no cover - GUI worker path
            self.signals.failed.emit(self.chapter.id, str(exc))


class PDFViewer(QtWidgets.QMainWindow):
    def __init__(
        self,
        pdf_path: Path,
        chapters: List[Chapter],
        explainer: Optional[GeminiExplainer],
        max_chars: int,
        context_chars: int,
        max_image_pixels: int,
    ) -> None:
        super().__init__()
        self.pdf_path = pdf_path
        self.doc = fitz.open(pdf_path)
        self.chapters = chapters
        self.explainer = explainer
        self.max_chars = max_chars
        self.context_chars = context_chars
        self.max_image_pixels = max_image_pixels
        self.thread_pool = QtCore.QThreadPool(self)
        self.thread_pool.setMaxThreadCount(2)
        self.chapter_requests: set[str] = set()
        self.current_page = 0
        self.current_zoom = 1.0
        self._suppress_chapter_event = False

        if self.explainer:
            for chapter in self.chapters:
                cached = self.explainer.get_cached(chapter)
                if cached and not chapter.summary:
                    chapter.summary = cached

        self._init_ui()
        self.setWindowTitle(f"Gemini PDF 해설 뷰어 - {pdf_path.name}")
        self.resize(1400, 900)
        self.statusBar().showMessage("PDF 로드 완료")
        self.show_page(0)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # pragma: no cover - GUI hook
        self.thread_pool.waitForDone(1000)
        return super().closeEvent(event)

    def _init_ui(self) -> None:
        central = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(central)

        # Left side: PDF page display
        left = QtWidgets.QVBoxLayout()

        self.page_label = QtWidgets.QLabel()
        self.page_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.page_label.setBackgroundRole(QtGui.QPalette.ColorRole.Base)
        self.page_label.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding)

        self.scroll_area = QtWidgets.QScrollArea()
        self.scroll_area.setWidget(self.page_label)
        self.scroll_area.setWidgetResizable(True)
        left.addWidget(self.scroll_area)

        nav_layout = QtWidgets.QHBoxLayout()
        self.prev_btn = QtWidgets.QPushButton("◀ 이전")
        self.next_btn = QtWidgets.QPushButton("다음 ▶")
        self.page_spin = QtWidgets.QSpinBox()
        self.page_spin.setMinimum(1)
        self.page_spin.setMaximum(self.doc.page_count)
        self.page_spin.setValue(1)
        self.zoom_combo = QtWidgets.QComboBox()
        for value in [50, 75, 100, 125, 150, 200]:
            self.zoom_combo.addItem(f"{value}%", value / 100)
        self.zoom_combo.setCurrentIndex(2)  # 100%

        nav_layout.addWidget(self.prev_btn)
        nav_layout.addWidget(self.next_btn)
        nav_layout.addWidget(QtWidgets.QLabel("페이지"))
        nav_layout.addWidget(self.page_spin)
        nav_layout.addWidget(QtWidgets.QLabel("배율"))
        nav_layout.addWidget(self.zoom_combo)
        nav_layout.addStretch()
        left.addLayout(nav_layout)

        layout.addLayout(left, 3)

        # Right side: chapter list + explanation
        right = QtWidgets.QVBoxLayout()

        self.chapter_list = QtWidgets.QListWidget()
        for chapter in self.chapters:
            if chapter.start_page == chapter.end_page:
                label = f"p{chapter.start_page + 1}: {chapter.title}"
            else:
                label = f"p{chapter.start_page + 1}-{chapter.end_page + 1}: {chapter.title}"
            self.chapter_list.addItem(label)
        if self.chapters:
            self.chapter_list.setCurrentRow(0)
        right.addWidget(QtWidgets.QLabel("페이지 선택"))
        right.addWidget(self.chapter_list, 1)

        self.summary_title = QtWidgets.QLabel("페이지 해설")
        self.summary_title.setStyleSheet("font-weight: bold; font-size: 16px;")
        right.addWidget(self.summary_title)

        self.summary_text = QtWidgets.QTextBrowser()
        self.summary_text.setOpenExternalLinks(True)
        right.addWidget(self.summary_text, 2)

        self.loading_bar = QtWidgets.QProgressBar()
        self.loading_bar.setRange(0, 0)
        self.loading_bar.hide()
        right.addWidget(self.loading_bar)

        self.refresh_button = QtWidgets.QPushButton("해설 새로 요청")
        right.addWidget(self.refresh_button)

        layout.addLayout(right, 2)

        self.setCentralWidget(central)

        # Connections
        self.prev_btn.clicked.connect(lambda: self.show_page(self.current_page - 1))
        self.next_btn.clicked.connect(lambda: self.show_page(self.current_page + 1))
        self.page_spin.valueChanged.connect(lambda value: self.show_page(value - 1))
        self.zoom_combo.currentIndexChanged.connect(self._on_zoom_changed)
        self.chapter_list.currentRowChanged.connect(self._on_chapter_selected)
        self.refresh_button.clicked.connect(self._on_refresh_requested)

    def _on_zoom_changed(self, index: int) -> None:
        zoom_value = self.zoom_combo.itemData(index)
        if zoom_value:
            self.current_zoom = float(zoom_value)
            self.show_page(self.current_page)

    def _on_chapter_selected(self, row: int) -> None:
        if row < 0 or row >= len(self.chapters):
            return
        if self._suppress_chapter_event:
            return
        self.show_page(self.chapters[row].start_page)

    def _on_refresh_requested(self) -> None:
        chapter = self.chapter_for_page(self.current_page)
        if chapter:
            chapter.summary = None
            print(f"[UI] '{chapter.title}' 해설을 새로 요청합니다.")
            self.statusBar().showMessage(f"'{chapter.title}' 해설 재요청 중...")
            self.request_summary(chapter, force=True)

    def show_page(self, page_index: int) -> None:
        if page_index < 0 or page_index >= self.doc.page_count:
            return
        self.current_page = page_index
        page = self.doc.load_page(page_index)
        zoom_matrix = fitz.Matrix(self.current_zoom, self.current_zoom)
        pix = page.get_pixmap(matrix=zoom_matrix, alpha=False)
        image = QtGui.QImage(
            pix.samples,
            pix.width,
            pix.height,
            pix.stride,
            QtGui.QImage.Format.Format_RGB888,
        )
        self.page_label.setPixmap(QtGui.QPixmap.fromImage(image))
        self.page_spin.blockSignals(True)
        self.page_spin.setValue(page_index + 1)
        self.page_spin.blockSignals(False)
        self._highlight_current_chapter()
        self._update_summary_panel()

    def _highlight_current_chapter(self) -> None:
        chapter_index = self._chapter_index_for_page(self.current_page)
        self._suppress_chapter_event = True
        if chapter_index is not None:
            self.chapter_list.setCurrentRow(chapter_index)
        self._suppress_chapter_event = False

    def _chapter_index_for_page(self, page_index: int) -> Optional[int]:
        for idx, chapter in enumerate(self.chapters):
            if chapter.contains(page_index):
                return idx
        return None

    def chapter_for_page(self, page_index: int) -> Optional[Chapter]:
        chapter_index = self._chapter_index_for_page(page_index)
        return self.chapters[chapter_index] if chapter_index is not None else None

    def _update_summary_panel(self) -> None:
        chapter = self.chapter_for_page(self.current_page)
        if not chapter:
            self.summary_title.setText("페이지 정보 없음")
            self.summary_text.setMarkdown("해당 페이지에 대한 해설 정보를 찾을 수 없습니다.")
            self._set_loading_state(False)
            return
        title = f"{chapter.title} (p{chapter.start_page + 1}-{chapter.end_page + 1})"
        self.summary_title.setText(title)
        if chapter.summary:
            self.summary_text.setMarkdown(chapter.summary)
            self._set_loading_state(False)
        elif not self.explainer:
            self.summary_text.setMarkdown(
                "Gemini 설정이 없어 해설을 불러올 수 없습니다. API 키를 설정하세요."
            )
            self._set_loading_state(False)
        else:
            self.summary_text.setMarkdown("Gemini 해설을 불러오는 중...")
            self._set_loading_state(True)
            self.request_summary(chapter)

    def request_summary(self, chapter: Chapter, force: bool = False) -> None:
        if not self.explainer:
            return
        if chapter.summary and not force:
            return
        if not force and chapter.id in self.chapter_requests:
            return

        worker = SummaryWorker(
            self.pdf_path,
            chapter,
            self.chapters,
            self.explainer,
            self.max_chars,
            self.context_chars,
            self.max_image_pixels,
        )
        worker.signals.completed.connect(self._on_summary_ready)
        worker.signals.failed.connect(self._on_summary_failed)
        self.chapter_requests.add(chapter.id)
        print(f"[UI] '{chapter.title}' 해설 요청 시작")
        if chapter.contains(self.current_page):
            self._set_loading_state(True)
            self.statusBar().showMessage(f"'{chapter.title}' 해설 불러오는 중...")
        self.thread_pool.start(worker)

    def prefetch_all_summaries(self) -> None:
        if not self.explainer:
            return
        print("[UI] 전체 페이지 해설을 백그라운드에서 준비합니다.")
        for chapter in self.chapters:
            if not chapter.summary:
                self.request_summary(chapter)

    def _on_summary_ready(self, chapter_id: str, summary: str) -> None:
        chapter = next((c for c in self.chapters if c.id == chapter_id), None)
        if not chapter:
            return
        chapter.summary = summary
        self.chapter_requests.discard(chapter_id)
        if chapter.contains(self.current_page):
            self.summary_text.setMarkdown(summary)
            self._set_loading_state(False)
            self.statusBar().showMessage(f"'{chapter.title}' 해설 업데이트 완료", 3000)

    def _on_summary_failed(self, chapter_id: str, error_message: str) -> None:
        self.chapter_requests.discard(chapter_id)
        if chapter_id == getattr(self.chapter_for_page(self.current_page), "id", None):
            self.summary_text.setMarkdown(f"해설 요청 실패: {error_message}")
            self._set_loading_state(False)
            self.statusBar().showMessage("해설 요청 실패", 5000)

    def _set_loading_state(self, is_loading: bool) -> None:
        self.loading_bar.setVisible(is_loading)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gemini 기반 PDF 해설 뷰어")
    parser.add_argument("pdf", type=Path, nargs="?", help="열고 싶은 PDF 경로")
    parser.add_argument(
        "--api-key",
        default=os.getenv("GOOGLE_API_KEY", ""),
        help="Gemini API 키. 기본값은 GOOGLE_API_KEY 환경 변수",
    )
    parser.add_argument(
        "--model",
        default="gemini-2.5-flash",
        help="요약에 사용할 Gemini 모델 이름",
    )
    parser.add_argument(
        "--max-text-chars",
        type=int,
        default=20000,
        help="각 페이지(전체 텍스트 포함)에 대해 Gemini로 보낼 최대 글자 수",
    )
    parser.add_argument(
        "--context-text-chars",
        type=int,
        default=4000,
        help="이전/다음 페이지 문맥으로 사용할 최대 텍스트 길이",
    )
    parser.add_argument(
        "--max-image-pixels",
        type=int,
        default=262144,
        help="이미지를 전송하기 전에 유지할 최대 픽셀 수(예: 512x512)",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(".cache"),
        help="Gemini 응답 캐시를 저장할 디렉터리",
    )
    parser.add_argument(
        "--skip-gemini",
        action="store_true",
        help="UI만 실행하고 Gemini 호출은 건너뜁니다.",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="지원되는 Gemini 모델 목록을 출력하고 종료",
    )
    return parser.parse_args()


def fetch_supported_models(api_key: str) -> List[str]:
    if not api_key:
        raise ValueError("모델 목록을 보려면 Gemini API 키가 필요합니다.")
    genai.configure(api_key=api_key)
    try:
        models = [
            model.name
            for model in genai.list_models()
            if "generateContent" in getattr(model, "supported_generation_methods", [])
        ]
    except Exception as exc:
        raise RuntimeError(f"모델 목록 조회에 실패했습니다: {exc}") from exc
    return sorted(models)


def pick_pdf_via_dialog() -> Optional[Path]:
    ensure_qapp()
    file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
        None,
        "열고 싶은 PDF 선택",
        str(Path.cwd()),
        "PDF Files (*.pdf)"
    )
    if not file_path:
        return None
    return Path(file_path)

def main() -> None:
    args = parse_args()
    if args.list_models:
        try:
            models = fetch_supported_models(args.api_key)
        except (ValueError, RuntimeError) as exc:
            raise SystemExit(str(exc))
        if not models:
            print("generateContent을 지원하는 모델을 찾지 못했습니다.")
        else:
            print("generateContent 지원 모델:")
            for name in models:
                print(f" - {name}")
        return

    if not args.pdf:
        selected = pick_pdf_via_dialog()
        if not selected:
            raise SystemExit("PDF를 선택하지 않아 종료합니다.")
        args.pdf = selected
    if not args.pdf.exists():
        raise SystemExit(f"PDF 파일을 찾을 수 없습니다: {args.pdf}")

    pdf_hash = hash_file(args.pdf)
    with fitz.open(args.pdf) as doc:
        chapters = build_chapters(doc)

    explainer: Optional[GeminiExplainer] = None
    cache_path = args.cache_dir / f"{args.pdf.stem}_summaries.json"

    if not args.skip_gemini:
        try:
            explainer = GeminiExplainer(
                api_key=args.api_key,
                model_name=args.model,
                pdf_hash=pdf_hash,
                cache_path=cache_path,
            )
        except ValueError as exc:
            raise SystemExit(str(exc))
        print("Gemini 해설을 백그라운드에서 준비합니다. UI는 즉시 표시됩니다.")
    else:
        print("Gemini 호출 없이 뷰어만 실행합니다.")

    app = ensure_qapp()
    viewer = PDFViewer(
        args.pdf,
        chapters,
        explainer,
        args.max_text_chars,
        args.context_text_chars,
        args.max_image_pixels,
    )
    viewer.show()
    if explainer:
        QtCore.QTimer.singleShot(0, viewer.prefetch_all_summaries)
    app.exec()


if __name__ == "__main__":
    main()
