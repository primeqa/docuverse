import sys


class Ticker:
    def __init__(self, message=None, step=7):
        """
        Initialize the Ticker object with optional message and step
        """
        self.message = message
        self.step = step
        self.line = 0
        self.prev_line = ""
        self.displayed = False
        # Ensure stderr is unbuffered
        sys.stderr.flush()

    def tick(self, force=None, new_value=None):
        """
        Updates the ticker value and prints to stderr based on conditions.
        """
        if (self.line >> self.step) > ((self.line - 1) >> self.step) or force is not None:
            if not self.displayed:
                # Print message for the first time
                if self.message:
                    sys.stderr.write(self.message)
                self.displayed = True
            else:
                # Clear the previous line
                sys.stderr.write("\b" * len(self.prev_line))
                sys.stderr.write(" " * len(self.prev_line))
                sys.stderr.write("\b" * len(self.prev_line))

            # Determine what to print
            if new_value is not None:
                str_val = str(new_value)
            else:
                str_val = self.commify(str(self.line))

            # Print the new value
            sys.stderr.write(str_val)
            self.prev_line = str_val
            sys.stderr.flush()

        # Increment the value if it is an integer
        if isinstance(self.line, int):
            self.line += 1

    def clear(self):
        """
        Clears the current ticker display from stderr.
        """
        length = len(self.prev_line) + (len(self.message) if self.message else 0)
        sys.stderr.write("\b" * length)
        sys.stderr.write(" " * length)
        sys.stderr.write("\b" * length)
        self.displayed = False
        self.line = 0
        self.prev_line = ""

    def set_message(self, new_message):
        """
        Set a new message for the ticker.
        """
        self.message = new_message

    def get_message(self):
        """
        Get the current message of the ticker.
        """
        return self.message

    def display_message(self):
        """
        Display the current message to stderr.
        """
        if self.message:
            sys.stderr.write(self.message)

    @staticmethod
    def commify(number):
        """
        Adds commas to a number for readability, e.g., '1000' -> '1,000'.
        """
        number = str(number)[::-1]
        chunks = [number[i:i + 3] for i in range(0, len(number), 3)]
        return ",".join(chunks)[::-1]


if __name__ == "__main__":
    ticker = Ticker(message="Processing: ", step=1)
    from time import sleep
    for i in range(100):
        ticker.tick()
        sleep(0.1)
        if i == 50:
            ticker.set_message("Halfway there! ")
        # if i % 10 == 0:
        #     sys.stderr.write("\n")
    ticker.clear()
