import math


class Point:
    def __init__(self, x=None, y=None):
        self.x = int(x) if x else x
        self.y = int(y) if y else y


class Line:
    def __init__(self, pt1, pt2):
        self.pt1 = pt1
        self.pt2 = pt2

    @property
    def center(self):
        center_x = int((self.pt1.x + self.pt2.x) / 2)
        center_y = int((self.pt1.y + self.pt2.y) / 2)
        return Point(center_x, center_y)

    @staticmethod
    def loc2line(t):
        """
        (x1, y1, x2, y2) -> line object
        """
        return Line(Point(*t[:2]), Point(*t[2:]))

    def line2loc(self):
        """
        line object -> (x1, y1, x2, y2)
        """
        return self.pt1.x, self.pt1.y, self.pt2.x, self.pt2.y

    @property
    def k(self):
        if self.pt2.x != self.pt1.x:
            return (self.pt2.y - self.pt1.y) / (self.pt2.x - self.pt1.x)
        return 99999

    @property
    def b(self):
        if self.pt2.x != self.pt1.x:
            return (self.pt2.y * self.pt1.x - self.pt1.y * self.pt2.x) / (self.pt1.x - self.pt2.x)
        return 0

    @property
    def A(self):
        return self.pt1.y - self.pt2.y

    @property
    def B(self):
        return self.pt2.x - self.pt1.x

    @property
    def C(self):
        return self.pt1.x * self.pt2.y - self.pt2.x * self.pt1.y

    @property
    def point1(self):
        return self.pt1.x, self.pt1.y

    @property
    def point2(self):
        return self.pt2.x, self.pt2.y


def distance_pts(p1, p2):
    """
    两点之间的距离
    """
    return int(((p1.x - p2.x)**2 + (p1.y - p2.y)**2)**0.5)


def distance_pt2line(point, line):
    """
    点到直线距离
    """

    try:
        d = abs(line.A * point.x + line.B * point.y + line.C) / math.sqrt(line.A * line.A + line.B * line.B)
        return d
    except Exception as e:
        print(str(e))
