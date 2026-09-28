import matplotlib.pyplot as plt


def plot2D(X, fig_size: tuple[int, int]=(6, 6)) -> None:
    fig, ax = plt.subplots(1, 1, figsize=fig_size)
    ax.scatter(X[:, 0], X[:, 1], s=1)
    ax.grid()
    ax.set_facecolor("lightgrey")

def plot3D(X, fig_size: tuple[int, int]=(6, 6)) -> None:
    fig = plt.figure(figsize=fig_size)
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], s=1)
    ax.set_aspect('equal')
    ax.grid()
    ax.set_facecolor("lightgrey")