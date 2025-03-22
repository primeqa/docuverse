import os

class FoF:
    def __init__(self, file):
        self.file = file
        self.dir = ""
        if file.find("@") > 0:
            self.file, self.dir = self.file.split("@")
            if os.path.exists(os.path.join(self.dir, self.file)):
                self.file = os.path.join(self.dir, self.file)

            self.files = open(self.file).readlines()
        else:
            self.files = open(self.file).readlines()
        self.files = [os.path.join(self.dir, f.strip()) for f in self.files]

    def basefile(self, file):
        return file.replace(self.dir+"/", "")

    def __iter__(self):
        return iter(self.files)

    def __len__(self):
        return len(self.files)