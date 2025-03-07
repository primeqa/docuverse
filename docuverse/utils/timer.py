# *****************************************************************#
# (C) Copyright IBM Corporation 2019.                             #
#                                                                 #
# The source code for this program is not published or otherwise  #
# divested of its trade secrets, irrespective of what has been    #
# deposited with the U.S. Copyright Office.                       #
# *****************************************************************#
'''Provide timers for quick profiling.
Author: Radu Florian
raduf@us.ibm.com )'''
from datetime import datetime
import sys
from typing import Dict

msec = 1
sec = 1000
minute = 60 * sec
hour = 60 * minute
day = 24 * hour
week = 7 * day
vals = [week, day, hour, minute, sec, msec]
names = ['week', 'day', 'hour', 'minute', 'second', 'millisecond']


class timer(object):
    '''Class to quickly compute timing, in a hierarchical fashion. To use, create an instance with
    a name, then use the tm.add_timing("name"). The variable "name" can contain namespaces
     (i.e. separate by using '::').
     The timer gather all the timers in a common space, and one can display the overall timing,
     by using timer.display_timing(), outputting something like this:
    Name                                      Time (s)      Rel %      Abs %  Spd(kw/s)  Spd(kc/s)
    MentionDetector                               27.9s      100.0%       99.3%       3.8      18.9
      decode                                      11.4s       40.9%       40.7%       9.2      46.1
        post_processing                            0.7s        5.9%        2.4%     157.9     787.2
          voting                                   0.1s        0.3%        0.4%    1054.9    5258.4
        read_examples                              2.3s       19.7%        8.0%      46.8     233.4
          Creating_pytorch_structures              0.1s        0.8%        0.4%    1054.9    5258.4
          convert_examples_to_features             2.2s       98.0%        7.9%      47.8     238.3
          create_examples_from_iob                 0.1s        0.8%        0.4%    1054.9    5258.4
        bert_run                                   8.5s       74.4%       30.3%      12.4      61.9
        data_prep                                  0.1s        0.0%        0.4%    1054.9    5258.4
      init                                        16.4s       59.0%       58.6%       6.4      32.0
        preamble                                   0.1s        0.0%        0.4%    1054.9    5258.4
        read_iob                                   0.2s        1.0%        0.6%     616.9    3075.1
      post_decoding                                0.1s        0.1%        0.4%    1054.9    5258.4
        write_output                               0.1s      100.0%        0.4%    1054.9    5258.4
    '''
    timing = {}
    processed = 0

    def __init__(self, name="timer", disable=False):
        self._orig = datetime.now()
        self._last = datetime.now()
        self._time = datetime.now()
        self.name = name
        self.disable = disable

    def mark(self):
        '''Starts a particular timing event.'''
        self._last = self._time
        self._time = datetime.now()

    def mark_and_return_time(self):
        '''Computes the humanly readable time from the previous mark() and returns it.'''
        self.mark()
        return self.time_since_last_mark()

    def mark_and_return_milliseconds(self):
        '''Marks and computes the milliseconds since the last mark()
        Returns:
            the milliseconds since the last mark()'''
        self.mark()
        return self.milliseconds_since_last_mark()

    def mark_and_return_microseconds(self):
        '''Marks and computes the microseconds since the last mark()
        Returns:
            The microseconds since the last mark()'''
        self.mark()
        return self.microseconds_since_last_mark()

    def time_since_last_mark(self):
        '''Computes the time since the last mark (but does not mark()).
        Returns:
            The humanly readable time since the last mark()
        '''
        # return self._time_between(self._time, self._last)
        return self.time_from_milliseconds(self.milliseconds_since_last_mark())

    def time_since_beginning(self):
        '''Returns the humanly interpretable time since the timer was created.
        Returns:
            The humanly readable time since the timer was created.
        '''
        self.mark()
        return self.time_from_milliseconds(self.milliseconds_since_beginning(), 0)

    def milliseconds_since_last_mark(self):
        '''Returns the milliseconds since the last mark().
        Returns:
            Milliseconds since the last mark.
        '''
        return self._milliseconds_between(self._time, self._last)

    def microseconds_since_last_mark(self):
        ''' Computes the microseconds since the last mark.
        Returns:
            microseconds since last mark.
        '''
        return timer._microseconds_between(self._time, self._last)

    def milliseconds_since_beginning(self):
        ''' Computes milliseconds since the beginning.
        Returns:
            milliseconds since the beginning.'''
        self.mark()
        return self._milliseconds_between(self._time, self._orig)

    @staticmethod
    def _microseconds_between(t1, t2):
        '''Internal function to compute microseconds between two times.
        Returns:
            the microseconds between two times.
        '''
        delta = t1 - t2
        return 1000000 * delta.seconds + delta.microseconds

    @staticmethod
    def _milliseconds_between(t1, t2):
        '''Computes and returns the number of milliseconds between two times.
        Returns:
            Returns the number of miliseconds between two times.'''
        delta = t1 - t2
        return int(1000.0 * delta.seconds + delta.microseconds / 1000.0)

    @staticmethod
    def _time_between(t1, t2):
        '''Compute the humnly interpretable time between two times.
        Returns:
            The humanly interpretable time between two times.'''
        # return str(t1 - t2)
        return timer.time_from_milliseconds(t1-t2)

    @staticmethod
    def time_from_milliseconds(millis, micros=0):
        '''Compute humanly readable time from milliseconds and microseconds.
        Arguments:
            millis - the time amount in milliseconds
            micros - the time amount in microseconds (if needed)
        Return:
            a string representing the humanly readable time represented by the given milliseconds
            and microseconds
        '''
        res = ""
        for n, v in zip(names, vals):
            val = int(millis / v)
            if val > 0:
                if res != "":
                    res += ", "
                res += "{} {}".format(val, n + "s" if val > 1 else n)
            millis -= v * val
        if res == "":
            res = "{} microsecond(s)".format(micros)
        return res

    def add_timing(self, key, ms=-1):
        '''Marks and adds the time to the specified label. The label can have "::" substring -
        allowing for hierarchical time reporting.
        Arguments:
            - key - the label to which to add timing
            - ms  - the millisecods to add. If not provided, the current timer is mark()ed, and
                    the timing from the last mark is used.
        '''
        if self.disable:
            return
        if ms == -1:
            ms = self.mark_and_return_microseconds()/1000.0
        timer.static_add_timing(self.name, key, ms)


    @staticmethod
    def static_add_timing(name, key, val):
        '''Adds the given number of milliseconds given in val to the given key.
        Arguments:
            - key - the label to add the timing to
            - val - the number of milliseconds to add
        '''
        key = name + "::" + key
        if key not in timer.timing:
            timer.timing[key] = val
        else:
            timer.timing[key] += val

    @staticmethod
    def _compute_timing_tree(namelist=None, level=0):
        '''Internal function to compute the namespace timing tree.
        Arguments:
            - namelist - the list of lists to display at this level of the tree (the list is
                         full from the root to leaf).
            - level    - the level of the current display_timing
        '''
        if namelist is None:
            namelist = [t.split("::") for t in timer.timing.keys()]

        if len(namelist) == 1:
            return [{"key":"::".join(namelist[0][level:]), "children":[],
                     "time":timer.timing["::".join(namelist[0])], "percent": 100}]
        else:
            _keys = sorted(namelist, key=lambda val: val[level])
            children = []
            dlist = {}
            leaves = {}
            prefixes = {}
            for key in _keys:
                K = key[level]
                if len(key) > level+1:
                    prefixes["::".join(key[:level+1])] = 1
                    if K not in dlist:
                        dlist[K] = [key]
                    else:
                        dlist[K] += [key]
                else:
                    leaves[K] = key


            res = []

            for K, L in dlist.items():
                children = timer._compute_timing_tree(L, level + 1)
                level_key = "::".join(L[0][0:level+1])
                time = sum([v['time'] for v in children]) \
                    if level_key not in timer.timing else timer.timing[level_key]
                time = max(time, 0.01)
                for child in children:
                    child['percent'] = child['time']/time*100
                res.append({"key": K, "children": children, "time": time, "percent": 100})

            for K, L in leaves.items():
                level_key = "::".join(L[0:level+1])
                if "::".join(L) not in prefixes:
                    children = timer._compute_timing_tree([L], level)
                    res.append(children[0])

        return res

    @staticmethod
    def display_timing(totalms, level=0, stat_list=None, keys: dict[str, int]|None=None,
                       sorted_by:str|None=None,
                       reverse=False, output_stream=None):
        '''Static method that will print the hierarchical times for the labeled times.
        The speeds computed are in kilo-words/sec and kilo-chars/sec.
        Arguments:
            - totalms - the total number of milliseconds - used to compute the absolute % times.
            - level   - the level of the tree to display (default 0)
            - stat_list - filter for the labels
            - keys: dict[Str:Int] - a list of keys to break performance by (e.g., num_words, num_chars, etc.)
            # - num_words - the number of processed words; needed for words/s speed calculation
            # - num_chars - the number of processed characters; needed for kc/s speed calculation
            '''
        if keys is None:
            keys = {}
        if output_stream is None:
            output_stream = sys.stdout
        # def _display_tree(node, level, num_words=1, num_chars=1):
        def _display_tree(node, level, keys: dict[str, int]):
            def process_val(v):
                vv = 1000*v*1.0/ms
                if vv < 10000:
                    return f"{vv:10.1f}"
                elif vv < 1000000:
                    return f"{vv/1000.0:10.1f}k"
                elif v < 10000000000:
                    return f"{vv/1000000.0:10.1f}M"
                else:
                    return f"{vv/100000000.0:10.1f}Bs"
            ms = max(100, node["time"])
            secs = max(0.1, ms/1000.0)
            perc = node["percent"]
            okey = (" " * 2 * level) + node['key']

            # print("{:<40s}{:>10.1f}s {:>10.1f}% {:10.1f}%{:10.1f}{:10.1f}".
            #       format(okey, secs, perc, ms*100 / totalms, num_words * 1.0 / ms,
            #              num_chars * 1.0 / ms),
            #       file=output_stream)
            print("{:<40s}{:>10.1f}s {:>10.1f}% {:10.1f}%".
                  format(okey, secs, perc, ms*100 / totalms),
                  file=output_stream, end='')
            for k, v in keys.items():
                print(process_val(v), end='', file=output_stream)
            print(file=output_stream)
            if sorted_by == "%":
                node["children"] = sorted(node["children"], key=lambda val: val["percent"], reverse=reverse)
            elif sorted_by == "name":
                node["children"] = sorted(node["children"], key=lambda val: val["key"], reverse=reverse)

            for child in node["children"]:
                _display_tree(child, level+1, keys)

        tree = timer._compute_timing_tree(None, 0)

        if len(tree) > 1:
            tree = {"key": "Root", 'children': tree, 'percent': 100, 'time': totalms}
            for child in tree['children']:
                child['percent'] = child['time']*100/totalms
        else:
            tree = tree[0]

        # if num_chars*1000/totalms > 1000000: # kchars/s > 1000
        #     range='M'
        #     num_chars /= 1000
        #     num_words /= 1000
        # elif num_words*1000/totalms<100: # kchars/s < 0.1
        #     num_chars *= 1000
        #     num_words *= 1000
        #     range=''
        # else:
        #     range='k'

        print(f"{'Name':<40s} {'Time (s)':>10s}  {'Rel %':>10s}  {'Abs %':>10s}", file=output_stream, end='')
        for k, v in keys.items():
            print(f"{k+'/s':>10s}", file=output_stream, end='')
        print(file=output_stream)

        _display_tree(tree, level, keys)

    @staticmethod
    def test():
        tm = timer("Something")
        tm.add_timing("method1", 1000)
        tm.add_timing("method1::submethod1", 500)
        tm.add_timing("method1::submethod2", 300)
        tm.add_timing("method2::submethod1", 600)
        tm.add_timing("method2::submethod2", 600)
        tm.add_timing("method3::submethod1::subsubmethod1", 100)
        tm.add_timing("method3::submethod1::subsubmethod2", 100)
        tm.add_timing("method4", 2000)

        tm = timer("Something_2")
        tm.add_timing("method3::sumethod1::subsubmethod1", 100)
        tm.add_timing("method4::submethod1::subsubmethod1", 100)
        tm.add_timing("method4::submethod1::subsubmethod2", 100)
        tm.add_timing("method4::submethod1", 250)

        tm.display_timing(9000, keys={'pigs':100, 'chars':10000})


if __name__ == "__main__":
    timer.test()
