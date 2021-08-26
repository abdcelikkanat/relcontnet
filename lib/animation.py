import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class Animation:

    def __init__(self, timePoints, r=1, c=1, figSize=(10, 10), bgColor='white',
                 color='k', marker='.', markerSize=10, delay=1000, margin=[0.1, 0.1]):

        self.__timePoints = timePoints
        self.__numOfTimePoints = len(self.__timePoints)
        # self.__numOfNodes = self.__x.shape[1]
        # self.__dim = self.__x.shape[2]

        self.__r = r
        self.__c = c
        self.__figSize = figSize
        self.__bgColor = bgColor
        self.__color = color
        self.__marker = marker
        self.__markerSize = markerSize
        self.__numOfFigures = r*c
        self.__delay = delay
        self.__margin = margin

        self.__x = [None for _ in range(self.__numOfFigures)]
        self.__titles = ["" for _ in range(self.__numOfFigures)]

        __fig = None
        __ax = None

        # Determine the axis limits for the latent position figure
        self.__xLeft = [0 for _ in range(self.__numOfFigures)]
        self.__yBelow = [0 for _ in range(self.__numOfFigures)]
        self.__xRight = [0 for _ in range(self.__numOfFigures)]
        self.__yTop = [0 for _ in range(self.__numOfFigures)]

        self.__fig, self.__ax = plt.subplots(self.__r, self.__c, figsize=self.__figSize)
        self.__setBackground()

    def __setBackground(self):

        if self.__numOfFigures == 1:
            self.__ax.clear()
            self.__ax.set_facecolor(self.__bgColor)
            self.__ax.set_xlim(self.__xLeft[0] - self.__margin[0], self.__xRight[0] + self.__margin[0])
            self.__ax.set_ylim(self.__yBelow[0] - self.__margin[1], self.__yTop[0] + self.__margin[1])
            pass
        else:
            for i in range(self.__numOfFigures):
                self.__ax[i].clear()
                self.__ax[i].set_facecolor(self.__bgColor)
                self.__ax[i].set_xlim(self.__xLeft[i] - self.__margin[0], self.__xRight[i] + self.__margin[0])
                self.__ax[i].set_ylim(self.__yBelow[i] - self.__margin[1], self.__yTop[i] + self.__margin[1])

    def __animate(self, t):

        self.__setBackground()

        for i in range(self.__numOfFigures):
            if self.__titles[i] == "":
                self.__titles[i] = "Time (t={:.2f})".format(self.__timePoints[t])

        if self.__numOfFigures == 1:
            for i in range(self.__x[0].shape[1]):
                if type(self.__color) == list:
                    color = self.__color[i]
                else:
                    color = self.__color
                self.__ax.plot(self.__x[0][t, i, 0], self.__x[0][t, i, 1], linestyle="None",
                               color=color, marker=self.__marker, markersize=self.__markerSize, alpha=0.75)

            self.__ax.title.set_text(self.__titles[0] + " ( Time: {:.2f} )".format(self.__timePoints[t]))
        else:
            for f in range(self.__numOfFigures):
                for i in range(self.__x[0].shape[1]):
                    if type(self.__color) == list:
                        color = self.__color[i]
                    else:
                        color = self.__color
                    self.__ax[f].plot(self.__x[f][t, i, 0], self.__x[f][t, i, 1], linestyle="None",
                                      color=color, marker=self.__marker, markersize=self.__markerSize, alpha=0.75)

                self.__ax[f].title.set_text(self.__titles[f] + " ( Time: {:.2f} )".format(self.__timePoints[t]))

    def __save(self, filePath):

        self.__anim.save(filePath, writer='imagemagick', savefig_kwargs={"facecolor": self.__bgColor})

    def addData(self, x, index=0, title=""):

        if index >= self.__r * self.__c:
            raise ValueError("Index cannot be equal to or greater than {}".format(self.__numOfFigures))

        self.__x[index] = x
        self.__titles[index] = title

        self.__xLeft[index], self.__yBelow[index] = np.min(np.min(x, axis=0), axis=0)[0], np.min(np.min(x, axis=0), axis=0)[1]  # np.min(np.min(self._z, axis=1), axis=0)
        self.__xRight[index], self.__yTop[index] = np.max(np.max(x, axis=0), axis=0)[0], np.max(np.max(x, axis=0), axis=0)[1]  # np.max(np.max(self._z, axis=1), axis=0)


    def plot(self, filePath):

        self.__anim = FuncAnimation(self.__fig, self.__animate, init_func=self.__setBackground, repeat=False,
                                    frames=self.__numOfTimePoints, interval=self.__delay)
        self.__save(filePath)






