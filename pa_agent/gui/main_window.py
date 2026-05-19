"""Main application window for PA Agent."""
from __future__ import annotations

import logging
from typing import Any

from PyQt6.QtCore import QThread, pyqtSignal, QObject
from PyQt6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMenuBar,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)
from PyQt6.QtGui import QAction
from PyQt6.QtCore import Qt

from pa_agent.app_context import AppContext

logger = logging.getLogger(__name__)

# Zombie timeout in milliseconds (5 seconds)
_WORKER_JOIN_TIMEOUT_MS = 5000


# ── AI Worker ─────────────────────────────────────────────────────────────────

class _AnalysisWorker(QThread):
    """Runs TwoStageOrchestrator.submit() on a background thread.

    Signals
    -------
    finished(dict):
        Emitted with the stage2_decision dict on success (or empty dict on
        failure / cancellation).
    status_update(str):
        Emitted with human-readable progress text.
    reasoning_token(str, str):
        Emitted with (stage, token_chunk) for each reasoning token streamed.
        stage is "stage1" or "stage2".
    content_token(str, str):
        Emitted with (stage, token_chunk) for each content token streamed.
        stage is "stage1" or "stage2".
    stage_prompt_ready(str, str, str):
        Emitted with (stage, system_prompt, user_prompt) just before each
        API call, so the conversation tab can show what was sent.
    """

    finished = pyqtSignal(dict)
    record_ready = pyqtSignal(object)   # emits the full AnalysisRecord
    status_update = pyqtSignal(str)
    reasoning_token = pyqtSignal(str, str)   # (stage, chunk)
    content_token = pyqtSignal(str, str)     # (stage, chunk)
    stage_prompt_ready = pyqtSignal(str, str, str)  # (stage, system, user)
    stage2_files_ready = pyqtSignal(list)  # strategy .txt filenames for stage 2

    def __init__(
        self,
        orchestrator: Any,
        frame: Any,
        cancel_token: Any,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._orchestrator = orchestrator
        self._frame = frame
        self._cancel_token = cancel_token

    def run(self) -> None:
        from pa_agent.util.threading import OrchestratorEvent

        _EVENT_LABELS = {
            OrchestratorEvent.Stage1Started: "阶段一分析中…",
            OrchestratorEvent.Stage1Done: "阶段一完成",
            OrchestratorEvent.Stage2Started: "阶段二分析中…",
            OrchestratorEvent.Stage2Done: "阶段二完成",
            OrchestratorEvent.RecordSaved: "记录已保存",
            OrchestratorEvent.Cancelled: "已取消",
            OrchestratorEvent.Stage1Failed: "阶段一失败",
            OrchestratorEvent.Stage2Failed: "阶段二失败",
        }

        def on_event(event: OrchestratorEvent) -> None:
            label = _EVENT_LABELS.get(event, str(event))
            self.status_update.emit(label)

        def on_stage1_reasoning(chunk: str) -> None:
            self.reasoning_token.emit("stage1", chunk)

        def on_stage1_content(chunk: str) -> None:
            self.content_token.emit("stage1", chunk)

        def on_stage2_reasoning(chunk: str) -> None:
            self.reasoning_token.emit("stage2", chunk)

        def on_stage2_content(chunk: str) -> None:
            self.content_token.emit("stage2", chunk)

        def on_stage_prompt(stage: str, system: str, user: str) -> None:
            self.stage_prompt_ready.emit(stage, system, user)

        def on_stage2_files(files: list[str]) -> None:
            self.stage2_files_ready.emit(files)

        try:
            record = self._orchestrator.submit(
                self._frame,
                self._cancel_token,
                on_event,
                on_stage1_reasoning=on_stage1_reasoning,
                on_stage1_content=on_stage1_content,
                on_stage2_reasoning=on_stage2_reasoning,
                on_stage2_content=on_stage2_content,
                on_stage_prompt=on_stage_prompt,
                on_stage2_files=on_stage2_files,
            )
            decision = record.stage2_decision or {}
        except Exception as exc:  # noqa: BLE001
            logger.error("Analysis worker error: %s", exc, exc_info=True)
            decision = {}
            record = None  # type: ignore[assignment]

        if record is not None:
            self.record_ready.emit(record)
        self.finished.emit(decision)


# ── MainWindow ────────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    """Top-level workbench: chart + AI sidebar (analysis / raw / decision)."""

    def __init__(self, ctx: AppContext, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("PA Agent — Trading Terminal")
        self.resize(1440, 900)
        self._ctx = ctx
        self._worker: _AnalysisWorker | None = None
        self._cancel_token: Any = None
        self._analysis_in_progress = False
        self._switching = False
        self._free_chat_session: Any = None
        self._last_stage1_diagnosis: dict | None = None
        # RefreshLoop runs in its own QThread
        self._refresh_loop: Any = None
        self._refresh_thread: QThread | None = None
        self._setup_ui()
        self._connect_event_bus()
        self._start_refresh_loop()
        self._update_ai_mode_label()

    # ── UI construction ───────────────────────────────────────────────────────

    def _setup_ui(self) -> None:
        from pa_agent.gui.ai_sidebar import AISidebar

        _api_key = ""
        _exc_counter = getattr(self._ctx, "exc_counter", None)
        _settings = getattr(self._ctx, "settings", None)
        if _settings is not None:
            _api_key = getattr(_settings.provider, "api_key", "") or ""

        self._ai_sidebar = AISidebar(
            api_key=_api_key,
            exc_counter=_exc_counter,
            settings=_settings,
        )
        self._stream_panel = self._ai_sidebar.stream
        self._debug_widget = self._ai_sidebar.debug
        self._prompt_files_panel = self._ai_sidebar.prompt_files
        self._decision_panel = self._ai_sidebar.decision

        self._central = self._build_workbench()
        self.setCentralWidget(self._central)

        # ── Status bar ────────────────────────────────────────────────────────
        self._status_bar = QStatusBar()
        self.setStatusBar(self._status_bar)
        self._status_bar.showMessage("就绪")

        # ── Menu bar ──────────────────────────────────────────────────────────
        menu_bar: QMenuBar = self.menuBar()  # type: ignore[assignment]
        settings_menu = menu_bar.addMenu("设置")

        open_settings_action = QAction("打开设置…", self)
        open_settings_action.triggered.connect(self._open_settings_dialog)
        settings_menu.addAction(open_settings_action)

    def _build_workbench(self) -> QWidget:
        """Build chart + AI sidebar workbench."""
        from pa_agent.gui.chart_widget import ChartWidget

        tab = QWidget()
        outer_layout = QVBoxLayout(tab)
        outer_layout.setContentsMargins(8, 8, 8, 8)
        outer_layout.setSpacing(6)

        # ── Control bar ───────────────────────────────────────────────────────
        ctrl_layout = QHBoxLayout()
        ctrl_layout.setSpacing(8)

        # Symbol — editable combo (user can type any MT5 symbol)
        ctrl_layout.addWidget(QLabel("品种:"))
        self._symbol_combo = QComboBox()
        self._symbol_combo.setEditable(True)
        self._symbol_combo.addItems(["XAUUSD", "EURUSD", "GBPUSD", "USDJPY", "USDCHF", "XAGUSD"])
        # Restore last-used symbol from settings
        _last_symbol = "XAUUSD"
        _last_tf = "1h"
        _settings = getattr(self._ctx, "settings", None)
        if _settings is not None:
            _last_symbol = getattr(_settings.general, "last_symbol", "XAUUSD") or "XAUUSD"
            _last_tf = getattr(_settings.general, "last_timeframe", "1h") or "1h"
        self._symbol_combo.setCurrentText(_last_symbol)
        self._symbol_combo.setMinimumWidth(110)
        self._symbol_combo.lineEdit().setPlaceholderText("输入品种名…")
        ctrl_layout.addWidget(self._symbol_combo)

        # Timeframe
        ctrl_layout.addWidget(QLabel("周期:"))
        self._tf_combo = QComboBox()
        self._tf_combo.addItems(["1m", "5m", "15m", "1h", "4h", "1d"])
        self._tf_combo.setCurrentText(_last_tf)
        self._tf_combo.setMinimumWidth(60)
        ctrl_layout.addWidget(self._tf_combo)
        # Bar count
        ctrl_layout.addWidget(QLabel("K线数:"))
        self._bar_count_spin = QSpinBox()
        self._bar_count_spin.setRange(2, 5000)
        self._bar_count_spin.setValue(100)
        self._bar_count_spin.setMinimumWidth(70)
        ctrl_layout.addWidget(self._bar_count_spin)

        ctrl_layout.addStretch()

        self._submit_btn = QPushButton("提交分析")
        self._submit_btn.setObjectName("primaryButton")
        self._submit_btn.setMinimumWidth(100)
        self._submit_btn.clicked.connect(self._on_submit_analysis)
        ctrl_layout.addWidget(self._submit_btn)

        self._decision_badge = QLabel("")
        self._decision_badge.setObjectName("mutedLabel")
        ctrl_layout.addWidget(self._decision_badge)

        self._ai_mode_label = QLabel("")
        self._ai_mode_label.setObjectName("mutedLabel")
        ctrl_layout.addWidget(self._ai_mode_label)

        outer_layout.addLayout(ctrl_layout)

        status_row = QHBoxLayout()
        status_row.addStretch()
        self._last_refresh_ts: float = 0.0
        self._refresh_elapsed_label = QLabel("距上次刷新: —")
        self._refresh_elapsed_label.setObjectName("mutedLabel")
        status_row.addWidget(self._refresh_elapsed_label)

        from PyQt6.QtCore import QTimer as _QTimer
        self._elapsed_ticker = _QTimer(tab)
        self._elapsed_ticker.setInterval(1000)
        self._elapsed_ticker.timeout.connect(self._update_refresh_elapsed)
        self._elapsed_ticker.start()

        outer_layout.addLayout(status_row)

        workbench = QSplitter(Qt.Orientation.Horizontal)

        self._chart_widget = ChartWidget()
        self._chart_widget.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        workbench.addWidget(self._chart_widget)

        self._ai_sidebar.setMinimumWidth(400)
        workbench.addWidget(self._ai_sidebar)

        workbench.setStretchFactor(0, 3)
        workbench.setStretchFactor(1, 2)

        outer_layout.addWidget(workbench, stretch=1)

        # Initial button state
        self._update_submit_button_state()

        # Connect symbol/timeframe combo boxes to the switch handler
        self._symbol_combo.currentTextChanged.connect(
            lambda _: self._on_symbol_or_tf_changed(
                self._symbol_combo.currentText(), self._tf_combo.currentText()
            )
        )
        self._tf_combo.currentTextChanged.connect(
            lambda _: self._on_symbol_or_tf_changed(
                self._symbol_combo.currentText(), self._tf_combo.currentText()
            )
        )

        return tab

    def _connect_event_bus(self) -> None:
        """Wire EventBus signals to status bar and tab slots (if bus is ready)."""
        bus = self._ctx.event_bus
        if bus is None:
            return
        bus.status.connect(self._on_status_update)

    def _start_refresh_loop(self) -> None:
        """Start the RefreshLoop only when the data source is connected."""
        data_source = getattr(self._ctx, "data_source", None)
        buffer = getattr(self._ctx, "buffer", None)
        if data_source is None or buffer is None:
            logger.debug("RefreshLoop not started: data_source or buffer not available")
            return

        # Don't start if the data source hasn't connected yet
        if not getattr(data_source, "_connected", False):
            logger.info("Data source not connected — RefreshLoop deferred.")
            self._status_bar.showMessage("数据源未连接，请检查网络后重启程序")
            return

        from pa_agent.data.refresh_loop import RefreshLoop
        from pa_agent.util.threading import CancelToken

        settings = getattr(self._ctx, "settings", None)
        interval_ms = 1000
        n_bars = 200
        if settings is not None:
            interval_ms = getattr(settings.general, "refresh_interval_ms", 1000)
            n_bars = getattr(settings.general, "default_bar_count", 200)

        self._refresh_cancel_token = CancelToken()
        self._refresh_loop = RefreshLoop(
            data_source=data_source,
            buffer=buffer,
            n_bars=n_bars,
            interval_ms=interval_ms,
            cancel_token=self._refresh_cancel_token,
        )

        # Wire RefreshLoop signals
        self._refresh_loop.frame_ready.connect(self._on_refresh_frame_ready)
        self._refresh_loop.status_changed.connect(self._on_status_update)

        self._refresh_loop.start()
        logger.info("RefreshLoop started for %s %s",
                    getattr(data_source, "_symbol", "?"),
                    getattr(data_source, "_timeframe", "?"))

    # ── Slots ─────────────────────────────────────────────────────────────────

    def _on_status_update(self, text: str) -> None:
        """Update the status bar with subscription / analysis / data-delay text."""
        self._status_bar.showMessage(text)
        if self._analysis_in_progress:
            panel = getattr(self, "_stream_panel", None)
            if panel is not None:
                panel.on_analysis_progress(text)

    def _update_refresh_elapsed(self) -> None:
        """Update the 'distance from last refresh' label every second."""
        import time as _time
        label = getattr(self, "_refresh_elapsed_label", None)
        if label is None:
            return
        if self._last_refresh_ts == 0.0:
            label.setText("距上次刷新: —")
            return
        elapsed = int(_time.monotonic() - self._last_refresh_ts)
        if elapsed < 60:
            label.setText(f"距上次刷新: {elapsed}s")
        else:
            m, s = divmod(elapsed, 60)
            label.setText(f"距上次刷新: {m}m{s:02d}s")
        # Turn red if stale (> 10 seconds without update)
        if elapsed > 10:
            label.setStyleSheet("color: #f85149; font-size: 11px;")
        else:
            label.setObjectName("mutedLabel")
            label.setStyleSheet("")

    def _on_data_frame(self, frame: Any) -> None:
        """Forward a new KlineFrame to the chart widget (throttled by 30 Hz timer)."""
        self._chart_widget.set_frame(frame)

    def _on_refresh_frame_ready(self, bars: Any) -> None:
        """Handle frame_ready signal from RefreshLoop.

        Builds a KlineFrame directly from the bars returned by latest_snapshot()
        rather than reading back from the buffer, which avoids ordering issues
        caused by repeated appendleft() calls corrupting the buffer's deque.
        """
        if not bars:
            return

        try:
            from pa_agent.data.snapshot import compute_indicators
            from pa_agent.data.base import KlineBar, KlineFrame
            from pa_agent.util.timefmt import now_local_ms
            import time as _time

            settings = getattr(self._ctx, "settings", None)
            n_bars = 200
            if settings is not None:
                n_bars = getattr(settings.general, "default_bar_count", 200)

            symbol = self._symbol_combo.currentText().strip()
            timeframe = self._tf_combo.currentText()

            # Use the bars directly from latest_snapshot (already newest-first,
            # bars[0] is the forming bar).  Re-assign seq numbers to guarantee
            # the bijection invariant expected by the rest of the system.
            raw = bars[:n_bars]
            if len(raw) < n_bars:
                # Not enough bars yet — skip this tick silently
                return

            rebased: list[KlineBar] = [
                KlineBar(
                    seq=i + 1,
                    ts_open=b.ts_open,
                    open=b.open,
                    high=b.high,
                    low=b.low,
                    close=b.close,
                    volume=b.volume,
                    closed=(i != 0),
                )
                for i, b in enumerate(raw)
            ]

            indicators = compute_indicators(rebased)
            frame = KlineFrame(
                symbol=symbol,
                timeframe=timeframe,
                bars=tuple(rebased),
                indicators=indicators,
                snapshot_ts_local_ms=now_local_ms(),
            )

            self._chart_widget.set_frame(frame)

            # Record the time of this successful chart update
            self._last_refresh_ts = _time.monotonic()
            self._update_refresh_elapsed()
        except Exception as exc:  # noqa: BLE001
            logger.debug("Frame build skipped: %s", exc)

    def _on_symbol_or_tf_changed(self, new_symbol: str, new_tf: str) -> None:
        """Handle symbol or timeframe combo box change.

        Steps (design §B.10, R3.1–R3.5):
        1. Cancel current AI worker and wait up to 5 s (zombie if timeout).
        2. Save partial record if analysis was in progress.
        3. Unsubscribe data source, clear buffer, re-subscribe.
        4. Reset ChartWidget.
        5. Destroy FreeChatSession, disable Tab2 input.
        6. Reset or preserve ledger based on settings.
        """
        if self._switching:
            return  # Prevent re-entrant calls

        self._switching = True
        try:
            # ── Step 1: Cancel current AI worker ─────────────────────────────
            if self._worker is not None and self._worker.isRunning():
                if self._cancel_token is not None:
                    self._cancel_token.set()
                finished = self._worker.wait(_WORKER_JOIN_TIMEOUT_MS)
                if not finished:
                    logger.warning(
                        "AI worker did not finish within %d ms after symbol/tf switch; "
                        "marking as zombie",
                        _WORKER_JOIN_TIMEOUT_MS,
                    )
                    # Mark as zombie — do not force-kill
                self._worker = None

            # ── Step 2: Save partial record if analysis was in progress ───────
            if self._analysis_in_progress:
                pending_writer = getattr(self._ctx, "pending_writer", None)
                if pending_writer is not None:
                    # We don't have the active record here; the orchestrator
                    # handles save_partial via the cancel token path.
                    # This is a belt-and-suspenders call for any record that
                    # may have been built but not yet saved.
                    try:
                        pending_writer.save_partial(None, reason="user_switched")
                    except Exception:  # noqa: BLE001
                        pass
                self._analysis_in_progress = False
                self._update_submit_button_state()

            # ── Step 3: Unsubscribe, clear buffer, re-subscribe ───────────────
            data_source = getattr(self._ctx, "data_source", None)
            buffer = getattr(self._ctx, "buffer", None)
            if data_source is not None:
                try:
                    data_source.unsubscribe()
                except Exception as exc:  # noqa: BLE001
                    logger.warning("unsubscribe failed: %s", exc)
            if buffer is not None:
                buffer.clear()
            if data_source is not None:
                try:
                    data_source.subscribe(new_symbol, new_tf)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("subscribe(%s, %s) failed: %s", new_symbol, new_tf, exc)

            # ── Step 4: Reset ChartWidget ─────────────────────────────────────
            if hasattr(self, "_chart_widget"):
                self._chart_widget.reset()

            # ── Step 5: Destroy FreeChatSession, disable Tab2 input ───────────
            self._free_chat_session = None
            self._disable_chat_input()

            # ── Step 6: Reset ledger (always reset on symbol/tf switch) ───────
            ledger = getattr(self._ctx, "ledger", None)
            if ledger is not None:
                try:
                    ledger.reset()
                except Exception as exc:  # noqa: BLE001
                    logger.debug("ledger.reset() failed: %s", exc)

            self._status_bar.showMessage(f"已切换至 {new_symbol} {new_tf}")
            logger.info("Symbol/TF switched to %s %s", new_symbol, new_tf)

            # Persist last-used symbol/timeframe to settings
            settings = getattr(self._ctx, "settings", None)
            if settings is not None:
                settings.general.last_symbol = new_symbol
                settings.general.last_timeframe = new_tf
                try:
                    from pa_agent.config.settings import save_settings
                    save_settings(settings)
                except Exception as exc:  # noqa: BLE001
                    logger.debug("Failed to persist symbol/tf to settings: %s", exc)

        finally:
            self._switching = False

    def _disable_chat_input(self) -> None:
        """Disable free-chat input in the AI stream window."""
        panel = getattr(self, "_stream_panel", None)
        if panel is not None:
            panel.set_input_enabled(False)

    def _on_submit_analysis(self) -> None:
        """Handle the '提交分析' button click."""
        if not self._can_submit():
            return

        # Cancel any existing worker before starting a new one
        if self._worker is not None and self._worker.isRunning():
            if self._cancel_token is not None:
                self._cancel_token.set()
            self._worker.wait(_WORKER_JOIN_TIMEOUT_MS)
            self._worker = None

        # Gather inputs
        symbol = self._symbol_combo.currentText()
        timeframe = self._tf_combo.currentText()
        bar_count = self._bar_count_spin.value()

        # Try to build a KlineFrame snapshot
        frame = self._take_snapshot(symbol, timeframe, bar_count)
        if frame is None:
            self._status_bar.showMessage("数据不足，请等待缓冲区填满后再提交")
            return

        # Build orchestrator (if ctx has the necessary components)
        orchestrator = self._build_orchestrator()
        if orchestrator is None:
            self._status_bar.showMessage("编排器未就绪，请检查设置")
            return

        # Create cancel token
        from pa_agent.util.threading import CancelToken

        self._cancel_token = CancelToken()

        # Start worker in its own QThread (worker IS a QThread subclass)
        self._worker = _AnalysisWorker(
            orchestrator=orchestrator,
            frame=frame,
            cancel_token=self._cancel_token,
            parent=None,
        )
        self._worker.finished.connect(self._on_analysis_finished)
        self._worker.record_ready.connect(self._on_record_ready)
        self._worker.status_update.connect(self._on_status_update)
        self._worker.finished.connect(lambda _: self._on_worker_done())

        panel = getattr(self, "_stream_panel", None)
        if panel is not None:
            self._worker.stage_prompt_ready.connect(panel.on_stage_prompt_ready)
            self._worker.reasoning_token.connect(panel.on_reasoning_token)
            self._worker.content_token.connect(panel.on_content_token)

        self._analysis_in_progress = True
        self._update_submit_button_state()
        self._status_bar.showMessage("分析中…")
        self._decision_badge.setText("分析中…")
        self._ai_sidebar.focus_stream()

        panel = getattr(self, "_stream_panel", None)
        if panel is not None:
            panel.clear()
            panel.on_analysis_started()
        debug = getattr(self, "_debug_widget", None)
        if debug is not None:
            debug.clear()

        pf = getattr(self, "_prompt_files_panel", None)
        if pf is not None:
            from pa_agent.ai.prompt_assembler import stage1_prompt_txt_files

            pf.clear()
            pf.set_stage1_files(stage1_prompt_txt_files())
            pf.set_extras(stage1_builtin=True)

        self._worker.stage2_files_ready.connect(
            self._on_stage2_files_ready,
            Qt.ConnectionType.UniqueConnection,
        )
        self._worker.start()

    def _on_stage2_files_ready(self, strategy_files: list) -> None:
        """Update 调试 tab when Stage 2 strategy .txt list is known."""
        pf = getattr(self, "_prompt_files_panel", None)
        if pf is None:
            return
        from pa_agent.ai.prompt_assembler import stage2_prompt_txt_files

        pf.set_stage2_files(stage2_prompt_txt_files(strategy_files))
        pf.set_extras(stage1_builtin=True, stage2_builtin=True)

    def _on_analysis_finished(self, decision: dict) -> None:
        """Called on the main thread when the AI worker completes.

        *decision* is the full stage2 JSON dict (``{"decision": {...},
        "diagnosis_summary": {...}}``).  The chart and panel widgets expect
        the inner ``decision`` sub-dict, so we extract it here.
        """
        if decision:
            inner = decision.get("decision", decision)
            self._chart_widget.set_decision(inner)
            self._decision_panel.set_decision(
                inner,
                diagnosis_summary=decision.get("diagnosis_summary"),
                stage1_diagnosis=self._last_stage1_diagnosis,
            )
            order = inner.get("order_type", "—")
            self._decision_badge.setText(f"决策: {order}")
        else:
            self._decision_panel.clear()
            self._decision_badge.setText("")

    def _on_record_ready(self, record: Any) -> None:
        """Push the full AnalysisRecord to the conversation and debug tabs."""
        import json as _json

        # ── Debug tab: add Stage1 and Stage2 turns ────────────────────────────
        debug = getattr(self, "_debug_widget", None)
        if debug is not None:
            # Stage 1 turn
            s1_msgs = getattr(record, "stage1_messages", []) or []
            s1_system = next((m.get("content", "") for m in s1_msgs if m.get("role") == "system"), "")
            s1_user = next((m.get("content", "") for m in s1_msgs if m.get("role") == "user"), "")
            s1_raw = getattr(record, "stage1_response", {}) or {}
            s1_diag = getattr(record, "stage1_diagnosis", None)
            s1_validation = _json.dumps(s1_diag, ensure_ascii=False, indent=2) if s1_diag else "（验证失败或无数据）"
            debug.add_turn({
                "label": "Stage1 诊断",
                "system_prompt": s1_system,
                "user_prompt": s1_user,
                "raw_response": s1_raw,
                "validation_info": s1_validation,
            })

            # Stage 2 turn
            s2_msgs = getattr(record, "stage2_messages", []) or []
            s2_system = next((m.get("content", "") for m in s2_msgs if m.get("role") == "system"), "")
            s2_user = next((m.get("content", "") for m in s2_msgs if m.get("role") == "user"), "")
            s2_raw = getattr(record, "stage2_response", {}) or {}
            s2_decision = getattr(record, "stage2_decision", None)
            s2_validation = _json.dumps(s2_decision, ensure_ascii=False, indent=2) if s2_decision else "（验证失败或无数据）"
            debug.add_turn({
                "label": "Stage2 决策",
                "system_prompt": s2_system,
                "user_prompt": s2_user,
                "raw_response": s2_raw,
                "validation_info": s2_validation,
            })

            # Exception info if any
            exc_info = getattr(record, "exception", None)
            if exc_info:
                debug.add_turn({
                    "label": "⚠ 异常",
                    "system_prompt": "",
                    "user_prompt": "",
                    "raw_response": {},
                    "validation_info": _json.dumps(exc_info, ensure_ascii=False, indent=2),
                })

        pf = getattr(self, "_prompt_files_panel", None)
        if pf is not None:
            from pa_agent.ai.prompt_assembler import (
                stage1_prompt_txt_files,
                stage2_prompt_txt_files,
            )

            strategy = getattr(record, "strategy_files_used", None) or []
            experience = getattr(record, "experience_loaded", None) or []
            pf.set_latest_run(
                stage1_prompt_txt_files(),
                stage2_prompt_txt_files(strategy),
                experience_count=len(experience),
            )

        s1_diag = getattr(record, "stage1_diagnosis", None) or {}
        # Cache for _on_analysis_finished (which fires after this)
        self._last_stage1_diagnosis = s1_diag if isinstance(s1_diag, dict) else None
        s2_full = getattr(record, "stage2_decision", None)
        if s2_full:
            inner = s2_full.get("decision", s2_full)
            self._decision_panel.set_decision(
                inner,
                diagnosis_summary=s2_full.get("diagnosis_summary"),
                stage1_diagnosis=s1_diag if isinstance(s1_diag, dict) else None,
            )

        panel = getattr(self, "_stream_panel", None)
        if panel is not None:
            s1_diag = getattr(record, "stage1_diagnosis", None)
            if s1_diag:
                s1_content = _json.dumps(s1_diag, ensure_ascii=False, indent=2)
                s1_raw = getattr(record, "stage1_response", {}) or {}
                s1_reasoning = ""
                if isinstance(s1_raw, dict):
                    choices = s1_raw.get("choices", [])
                    if choices:
                        msg = choices[0].get("message", {})
                        s1_reasoning = msg.get("reasoning_content", "") or ""
                panel.show_stage_result("阶段一：市场诊断", s1_content, s1_reasoning)

            s2_decision = getattr(record, "stage2_decision", None)
            if s2_decision:
                s2_content = _json.dumps(s2_decision, ensure_ascii=False, indent=2)
                s2_raw = getattr(record, "stage2_response", {}) or {}
                s2_reasoning = ""
                if isinstance(s2_raw, dict):
                    choices = s2_raw.get("choices", [])
                    if choices:
                        msg = choices[0].get("message", {})
                        s2_reasoning = msg.get("reasoning_content", "") or ""
                panel.show_stage_result("阶段二：交易决策", s2_content, s2_reasoning)

            # ── Create FreeChatSession and wire to stream panel ───────────────
            try:
                from pa_agent.orchestrator.free_chat import FreeChatSession
                from pa_agent.util.threading import CancelToken as _CancelToken

                client = getattr(self._ctx, "client", None)
                assembler = getattr(self._ctx, "assembler", None)
                pending_writer = getattr(self._ctx, "pending_writer", None)
                ledger = getattr(self._ctx, "ledger", None)
                settings = getattr(self._ctx, "settings", None)

                if all(x is not None for x in [client, assembler, pending_writer, ledger]):
                    # Build a snapshot function that returns the latest closed K-line data
                    kline_snapshot_fn = self._make_kline_snapshot_fn()

                    session = FreeChatSession(
                        base_record=record,
                        client=client,
                        assembler=assembler,
                        pending_writer=pending_writer,
                        ledger=ledger,
                        settings=settings,
                        kline_snapshot_fn=kline_snapshot_fn,
                    )
                    chat_cancel_token = _CancelToken()
                    panel.set_session(session, chat_cancel_token)
                    logger.info("FreeChatSession created for record %s", getattr(record.meta, "timestamp_local_iso", "?"))
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to create FreeChatSession: %s", exc)

            panel.on_record_saved()

            usage_total = getattr(record, "usage_total", {}) or {}
            if usage_total:
                settings = getattr(self._ctx, "settings", None)
                context_window = 1_000_000
                if settings is not None:
                    context_window = getattr(settings.provider, "context_window", 1_000_000) or 1_000_000

                prompt_tokens = usage_total.get("prompt_tokens", 0)
                cached_tokens = usage_total.get("cached_prompt_tokens", 0)
                completion_tokens = usage_total.get("completion_tokens", 0)
                total_tokens = usage_total.get("total_tokens", 0) or (prompt_tokens + completion_tokens)

                panel.update_token_display({
                    "context_used": total_tokens,
                    "context_window": context_window,
                    "total_input": prompt_tokens,
                    "total_cached_input": cached_tokens,
                    "total_output": completion_tokens,
                })

    def _on_worker_done(self) -> None:
        """Reset in-progress flag and re-enable the submit button."""
        self._analysis_in_progress = False
        self._worker = None
        self._update_submit_button_state()
        self._status_bar.showMessage("分析完成")

    def _open_settings_dialog(self) -> None:
        """Open the SettingsDialog; import lazily to avoid circular imports."""
        from pa_agent.gui.settings_dialog import SettingsDialog
        from pa_agent.config.settings import Settings

        settings: Settings = self._ctx.settings  # type: ignore[assignment]
        if settings is None:
            settings = Settings()

        dlg = SettingsDialog(settings, parent=self)
        if dlg.exec():
            self._ctx.settings = settings
            client = getattr(self._ctx, "client", None)
            if client is not None:
                try:
                    client._settings = settings.provider  # type: ignore[attr-defined]
                except Exception:  # noqa: BLE001
                    pass
            if settings is not None:
                key = getattr(settings.provider, "api_key", "") or ""
                self._debug_widget._api_key = key
                self._ai_sidebar.bind_settings(settings)
            self._update_ai_mode_label()

    def _update_ai_mode_label(self) -> None:
        """Show current thinking / reasoning_effort / model in the toolbar."""
        settings = getattr(self._ctx, "settings", None)
        if settings is None:
            self._ai_mode_label.setText("")
            return
        p = settings.provider
        thinking = "开" if p.thinking else "关"
        self._ai_mode_label.setText(
            f"思考: {thinking} · effort={p.reasoning_effort} · {p.model}"
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _can_submit(self) -> bool:
        """Return True if the submit button should be enabled."""
        if self._analysis_in_progress:
            return False
        if self._switching:
            return False
        exc_count = self._get_consecutive_count()
        if exc_count >= 2:
            return False
        return True

    def _update_submit_button_state(self) -> None:
        """Enable or disable the submit button based on current state."""
        self._submit_btn.setEnabled(self._can_submit())

    def _get_consecutive_count(self) -> int:
        """Return the current consecutive exception count (0 if unavailable)."""
        try:
            exc_counter = getattr(self._ctx, "exc_counter", None)
            if exc_counter is not None:
                return exc_counter.consecutive_count
        except Exception:  # noqa: BLE001
            pass
        return 0

    def _take_snapshot(self, symbol: str, timeframe: str, bar_count: int) -> Any:
        """Snapshot for analysis: *bar_count* closed bars (newest forming bar excluded)."""
        try:
            from pa_agent.data.snapshot import build_analysis_frame

            data_source = getattr(self._ctx, "data_source", None)
            if data_source is None or not getattr(data_source, "_connected", False):
                return None

            bars_raw = data_source.latest_snapshot(bar_count + 5)
            if not bars_raw:
                return None

            return build_analysis_frame(bars_raw, bar_count, symbol, timeframe)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Snapshot failed: %s", exc)
            return None

    def _make_kline_snapshot_fn(self) -> Any:
        """Return a callable that captures the latest closed K-line data as a text table.

        The returned function reads from the live data source at call time,
        so FreeChatSession always gets the most recent market data when the
        user sends a follow-up message.
        """
        from pa_agent.ai.prompt_assembler import PromptAssembler

        symbol = self._symbol_combo.currentText()
        timeframe = self._tf_combo.currentText()
        bar_count = self._bar_count_spin.value()

        def _snapshot() -> str:
            frame = self._take_snapshot(symbol, timeframe, bar_count)
            if frame is None:
                return ""
            return PromptAssembler._render_kline_table(frame)

        return _snapshot

    def _build_orchestrator(self) -> Any:
        """Build a TwoStageOrchestrator from ctx components, or return None."""
        try:
            from pa_agent.orchestrator.two_stage import TwoStageOrchestrator

            client = getattr(self._ctx, "client", None)
            assembler = getattr(self._ctx, "assembler", None)
            router = getattr(self._ctx, "router", None)
            validator = getattr(self._ctx, "validator", None)
            exc_counter = getattr(self._ctx, "exc_counter", None)
            pending_writer = getattr(self._ctx, "pending_writer", None)
            exp_reader = getattr(self._ctx, "exp_reader", None)
            settings = getattr(self._ctx, "settings", None)

            if any(
                x is None
                for x in [client, assembler, router, validator, exc_counter,
                           pending_writer, exp_reader]
            ):
                return None

            return TwoStageOrchestrator(
                client=client,
                assembler=assembler,
                router=router,
                validator=validator,
                exc_counter=exc_counter,
                pending_writer=pending_writer,
                exp_reader=exp_reader,
                settings=settings,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not build orchestrator: %s", exc)
            return None
