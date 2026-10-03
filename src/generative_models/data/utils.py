import matplotlib.pyplot as plt
from matplotlib.axes._axes import Axes


def plot2D(X, ax: Axes | None = None, fig_size: tuple[int, int]=(6, 6), colour: str = "blue") -> Axes:
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=fig_size)
        ax.grid()
        ax.set_facecolor("lightgrey")
        ax.set_aspect("equal")

    ax.scatter(X[:, 0], X[:, 1], s=1, c=colour)
    return ax
    

def plot3D(X, fig_size: tuple[int, int]=(6, 6)) -> None:
    fig = plt.figure(figsize=fig_size)
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], s=1)
    ax.set_aspect('equal')
    ax.grid()
    ax.set_facecolor("lightgrey")
    ax.set_aspect("equal")