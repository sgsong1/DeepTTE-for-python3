import os
import datetime

class Logger:
    def __init__(self, exp_name):
        # 로그를 저장할 디렉토리 경로 설정
        log_dir = './logs'
        
        # 디렉토리가 없으면 생성
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        # 로그 파일을 생성
        self.file = open(f'{log_dir}/{exp_name}.log', 'w')

    def log(self, content):
        # 로그 내용을 파일에 기록하고 즉시 저장
        self.file.write(content + '\n')
        self.file.flush()
