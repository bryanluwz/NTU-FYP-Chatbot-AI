from datetime import datetime


class BaseModel:
    def __init__(self, debug=False):
        self.debug = debug

    def _debug_print(self, *msg):
        """
        Print debug messages
        """
        if self.debug:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S:%f")
            print(f"[{timestamp}]", *msg, end="\n\n")
