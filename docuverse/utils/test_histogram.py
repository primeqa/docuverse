from text_histogram import histogram
import random
histogram([random.gauss(50, 20) for _ in range(0,100)])