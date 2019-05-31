import numpy as np
import wave

class WaveFourierTransform:
    """
    waveファイルのフーリエ変換を行います

    Attributes
    ----------
    path: string
        このファイルのパス
    data: numpy.ndarray
        このファイルのフーリエ変換用のwaveバイナリ(未変換)
    rate: int
        このファイルのサンプリングレート
    frame: int
        このファイルのフレーム数 -> フレーム数 / サンプリングレート = このファイルの再生時間
    sampleList: [int][[numpy.ndarray], [float]]
        短時間フーリエ変換のフレーム別の周波数と振幅のリスト
        [フレーム数][fftData, freqList]
    nfft: int
        短時間フーリエ変換したときの1サンプルあたりのフレーム数
    overlap: int
        短時間フーリエ変換したときの重複しているフレーム数
    """
    def __init__(self, path):
        self.path = path
        wf = wave.open(path, "rb")
        self.data = np.frombuffer(wf.readframes(wf.getnframes()), dtype = 'int16')
        self.rate = wf.getframerate()
        self.frame = wf.getnframes()
        wf.close()
        self.sampleList = []
        self.nfft = None
        self.overlap = None

    def getAverage(self):
        """
        読み込んだデータ全体の振幅平均を返却

        Returns
        -------
        average: float
            このファイルの振幅平均
        """
        fftData, dummy = self.__fourierTransform(self.data)
        average = sum(fftData) / len(fftData)
        return average

    # 短時間フーリエ変換
    # https://qiita.com/namaozi/items/dec1575cd455c746f597
    def shortTimeFourierTransform(self, nfft):
        """
        読み込んだデータに対して短時間フーリエ変換を実施し、結果をインスタンス変数に格納する

        Prarameters
        -----------
        nfft: int
            1サンプルあたりのフーリエ変換フレーム数
        """
        self.nfft = int(nfft)
        # frame_length = self.data.shape[0] # wavファイルの全フレーム数
        frame_length = self.frame # self.data.shape[0]のちょうど半分なんだけど何だろう？
        self.overlap = self.nfft // 2 # 窓をずらした時のフレームの重なり具合. half shiftが一般的らしい
        time_song = float(frame_length) / self.rate  # 波形長さ(秒)
        time_unit = 1 / float(self.rate) # 1サンプルの長さ(秒)

        # FFTのフレームの時間を決めていきます
        # time_rulerに各フレームの中心時間が入っています
        start = (self.nfft / 2) * time_unit
        stop = time_song
        step =  (self.nfft - self.overlap) * time_unit
        time_ruler = np.arange(start, stop, step)

        # 窓関数は周波数解像度が高いハミング窓を用います
        window = np.hamming(self.nfft)
        spec = np.zeros([len(time_ruler), 1 + (self.nfft // 2)]) #転置状態で定義初期化
        pos = 0
        retList = []

        for fft_index in range(len(time_ruler)):
            frame = self.data[pos:pos + self.nfft]
            # フレームが信号から切り出せない時はアウトです
            if len(frame) == self.nfft:
                windowed = window * frame # ウィンドウ関数をかける
                fftData, freqList = self.__fourierTransform(windowed)
                retList.append([fftData, freqList])

                pos += (self.nfft - self.overlap)
        self.sampleList = retList

    def __fourierTransform(self, data):
        """
        データに応じたフーリエ変換を実施

        Parameters
        -----------
        data: numpy.ndarray
            フーリエ変換したいバイナリデータ

        Returns
        -------
        transformedList: [[numpy.ndarray], [float]]
            フーリエ変換後の周波数と振幅のリスト
            ([fftData], [freqList])
        """
        fftData = np.abs(np.fft.fft(data)) # 周波数帯域に応じた振幅
        freqList = np.fft.fftfreq(data.shape[0], d = 1.0 / self.rate) # 周波数帯域
        # 前半分しかいらないので後半分を削除する
        fftData = fftData[:len(fftData) // 2]
        freqList = freqList[:len(freqList) // 2]
        transformedList = fftData, freqList
        return transformedList

    def plot(self, fftData, freqList, minFreq = 0, maxFreq = 3000):
        """
        引数に渡したフーリエ変換データをプロットする

        Parameters
        ----------
        fftData: [numpy.ndarray]
            フーリエ変換したバイナリデータ
        freqList: [float]
            fftDataに対応した周波数リスト
        minFreq: float, default 0
            プロットする周波数の下限値
        maxFreq: float, default 3000
            プロットする周波数の上限値
        """
        import matplotlib.pyplot as plt
        plt.plot(freqList, fftData)
        plt.xlim(minFreq, maxFreq)
        plt.show()

    def plotByIndex(self, index, minFreq = 0, maxFreq = 3000):
        """
        最後に行った短時間フーリエ変換のindexサンプル目をプロット

        Parameters
        ----------
        index: int
            プロットしたいサンプルインデックス
        minFreq: float, default 0
            プロットする周波数の下限値
        maxFreq: float, default 3000
            プロットする周波数の上限値
        """
        if self.nfft == None:
            print("Please shortTimeFourierTransform before this method.")
            return
        import matplotlib.pyplot as plt
        plt.plot(self.sampleList[index][1], self.sampleList[index][0])
        plt.xlim(minFreq, maxFreq)
        plt.show()

    def __getMaxIndex(self, fftData, freqList):
        """
        引数に渡したフーリエ変換データのうち最も強い周波数帯のインデックスを返す

        Parameters
        ----------
        fftData: [numpy.ndarray]
            フーリエ変換したバイナリデータ
        freqList: [float]
            fftDataに対応した周波数リスト

        Returns
        -------
        maxIndex: int
            サンプルインデックス
        """
        maxIndex = list(fftData).index(fftData.max())
        # return fftData.max(), str(freqList[maxIndex])
        return maxIndex
        # print(index, fftData.max(), str(freqList[maxIndex]) + "Hz", "Index: " + str(maxIndex))
        # print("AVE", sum(fftData) / len(fftData))

    def debugPlot(self, minFreq = 0, maxFreq = 3000):
        fftData, freqList = self.__fourierTransform()
        maxIndex = list(fftData).index(fftData.max())
        print("MAX", fftData.max(), str(freqList[maxIndex]) + "Hz", "Index: " + str(maxIndex))
        print("AVE", sum(fftData) / len(fftData))
        self.plot(fftData, freqList, minFreq, maxFreq)
        return fftData, freqList

    def __isUnTransformed(self):
        if self.nfft == None:
            print("Please shortTimeFourierTransform before this method.")
            return True
        return False

    def getExceededIndexes(self, threshold):
        """
        最後に行った短時間フーリエ変換のうち、閾値を超えた振幅のインデックスのリストを各サンプルごとにリスト化して返却

        Parameters
        ----------
        threshold: float
            閾値

        Returns
        -------
        retList: [[int]]
            閾値を超えた周波数インデックスのリスト
        """
        if self.__isUnTransformed():
            return
        retList = []
        for fftData, freqList in self.sampleList:
            ampList = []
            index = 0
            for amp in fftData:
                if amp > threshold:
                    ampList.append(index)
                index += 1
            retList.append(ampList)
            # start = self.overlap * index / self.rate
            # end = (self.overlap * index + self.nfft) / self.rate
            # print(index, f"{start:.3f} - {end:.3f} secs.", fftData.max(), str(freqList[self.__getMaxIndex(fftData, freqList)]) + "Hz")
        return retList

    def getMaxList(self):
        """
        最後に行ったフーリエ変換のうち、各サンプルのもっとも強かった振幅と周波数をリストで返却

        Returns
        -------
        retList: [[float, float]]
            各サンプルでの一番強い振幅とその周波数のリスト
        """
        if self.__isUnTransformed():
            return
        retList = []
        for fftData, freqList in self.sampleList:
            index = self.__getMaxIndex(fftData, freqList)
            retList.append([fftData.max(), freqList[index]])
        return retList

    def getTimeBySampleIndex(self, index):
        """
        最後に行った短時間フーリエ変換のうち、指定したサンプルインデックスの開始秒と終了秒を返却

        Returns
        -------
        start: float
            対象となるサンプルの開始秒
        end: float
            対象となるサンプルの終了秒
        """
        start = self.overlap * index / self.rate
        end = (self.overlap * index + self.nfft) / self.rate
        return start, end

    def getExceededIndexOfSpecifyFreq(self, threshold, freqIndex):
        """
        最後に行った短時間フーリエ変換のうち、指定した周波数の振幅が指定した閾値を超えたサンプルインデックスリストを返却

        Parameters
        ----------
        threshold: float
            閾値
        freqIndex: int
            周波数のインデックス

        Returns
        -------
        retList: [int]
            振幅が閾値を超えた周波数のインデックス
        """
        if self.__isUnTransformed():
            return
        retList = []
        index = 0
        for fftData, freqList in self.sampleList:
            if fftData[freqIndex] > threshold:
                retList.append(index)
            index += 1
        return retList

    def getFreqIndex(self, selectFreq):
        """
        最後に行った短時間フーリエ変換のうち、指定した周波数に該当するフレームインデックスを返却

        Parameters
        ----------
        selectFreq: float
            インデックスを知りたい周波数

        Returns
        -------
        index: int
            当該周波数のインデックス
        """
        if self.__isUnTransformed():
            return
        if selectFreq <= 0:
            return None
        freqList = self.sampleList[0][1]
        index = 0
        for freq in freqList:
            if freq > selectFreq:
                return index - 1
            index += 1
        return index

if __name__ == "__main__":
    from sys import argv
    if len(argv) > 1:
        path = argv[1]
    else:
        print("Please enter wave path.")
        path = input()
    wft = WaveFourierTransform(path)

    if len(argv) > 2:
        nfft = wft.frame // int(argv[2])
    else:
        print("Please enter nfft")
        nfft = wft.frame // int(input())
    wft.shortTimeFourierTransform(nfft)

    average = wft.getAverage()
