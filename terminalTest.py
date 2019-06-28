import time
import curses

stdscr = curses.initscr()

stdscr.addstr(0, 0, "Hello")
stdscr.refresh()

time.sleep(1)

stdscr.addstr(0, 0, "World! (with curses)")
stdscr.refresh()