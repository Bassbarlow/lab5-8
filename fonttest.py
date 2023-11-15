from svgpath2mpl import parse_path
from svgpathtools import svg2paths
import matplotlib as mpl
import numpy as np; np.random.seed(32)
from matplotlib.path import Path
from matplotlib.textpath import TextToPath
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt



fig, ax = plt.subplots()


x = np.random.randn(4,10)
c = np.random.rand(10)
s = np.random.randint(120,500, size=10)
horse_path, attributes = svg2paths('horse.svg')
horse_marker = parse_path(attributes[0]['d'])

horse_marker.vertices -= horse_marker.vertices.mean(axis=0)
horse_marker = horse_marker.transformed(mpl.transforms.Affine2D().rotate_deg(180))
horse_marker = horse_marker.transformed(mpl.transforms.Affine2D().scale(-2,2))
plt.scatter(*x[:2], s=s, c=c, marker=horse_marker,
            edgecolors="none", linewidth=4)

plt.show()