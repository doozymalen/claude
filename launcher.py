"""실행 파일(.exe)과 run.bat에서 쓰는 시작 스크립트."""

import multiprocessing

from stele_extractor.gui.app import main

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
