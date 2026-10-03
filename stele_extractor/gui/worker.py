"""시간이 걸리는 계산을 화면이 멈추지 않도록 뒤에서 실행한다."""

from __future__ import annotations

import traceback

from PySide6.QtCore import QObject, QRunnable, QThreadPool, QTimer, Signal


class _Signals(QObject):
    done = Signal(object)
    failed = Signal(str, str)  # 사용자용 메시지, 자세한 내용
    progress = Signal(str)


# 실행 중인 작업을 붙잡아 둔다. 그렇지 않으면 결과 신호가 도착하기 전에
# 파이썬이 객체를 지워 버려 결과가 사라질 수 있다.
_ACTIVE: set = set()


class Task(QRunnable):
    def __init__(self, fn, *args, **kwargs):
        super().__init__()
        self.fn, self.args, self.kwargs = fn, args, kwargs
        self.signals = _Signals()
        self.setAutoDelete(False)

    def run(self):
        try:
            result = self.fn(*self.args, **self.kwargs)
        except Exception as e:  # noqa: BLE001 - 모든 오류를 화면에 알린다
            self.signals.failed.emit(str(e) or e.__class__.__name__, traceback.format_exc())
            return
        self.signals.done.emit(result)


def run_task(fn, *args, on_done=None, on_failed=None, **kwargs) -> Task:
    task = Task(fn, *args, **kwargs)
    _ACTIVE.add(task)
    if on_done:
        task.signals.done.connect(on_done)
    if on_failed:
        task.signals.failed.connect(on_failed)
    # 결과를 다 전달한 뒤에 놓아준다 (연결 순서대로 호출됨)
    task.signals.done.connect(lambda _: QTimer.singleShot(0, lambda: _ACTIVE.discard(task)))
    task.signals.failed.connect(lambda *_: QTimer.singleShot(0, lambda: _ACTIVE.discard(task)))
    QThreadPool.globalInstance().start(task)
    return task
