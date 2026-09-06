"""Command-line entry point for the analysis reports.

    mm-report            everything
    mm-report board      board statistics only
    mm-report matrix|head|sens|ai
"""
import sys

from match_monsters.analysis import reports


def main():
    reports.main()


if __name__ == '__main__':
    main()
