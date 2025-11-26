# source: https://github.com/EvoMindLab/EDOLAB/tree/main/Benchmark/GMPB
import numpy as np
import math
import copy
from metaevobox.environment.problem import Basic_Problem

class GMPB(Basic_Problem):
    def __init__(self,
                 dim=5,
                 PeakNumber=10,
                 ChangeFrequency=5000,
                 ShiftSeverity=1,
                 EnvironmentNumber=100,
                 HeightSeverity=7,
                 WidthSeverity=1,
                 AngleSeverity=math.pi / 9,
                 TauSeverity=0.2,
                 EtaSeverity=10):
        self.fes = 0
        self.T1 = [0]
        self.avg_dist = None
        self.initialize(dim, PeakNumber, ChangeFrequency, ShiftSeverity, EnvironmentNumber, HeightSeverity,
                        WidthSeverity, AngleSeverity, TauSeverity, EtaSeverity)
        self.optimum = self.OptimumValue[self.Environmentcounter]

    def __str__(self):
        return "GMPB"

    def reset(self):
        self.fes = 0
        self.T1 = [0]
        self.avg_dist = None
        self.optimum = self.OptimumValue[self.Environmentcounter]
        self.Environmentcounter = 0
        self.RecentChange = 0
        self.Ebbc = np.full(self.EnvironmentNumber, np.nan)
        self.CurrentError = np.full(self.maxfes + 1, np.nan)
        self.CurrentPerformance = np.full(self.maxfes + 1, np.nan)

    def initialize(self,
                   dim=5,
                   PeakNumber=10,
                   ChangeFrequency=5000,
                   ShiftSeverity=1,
                   EnvironmentNumber=100,
                   HeightSeverity=7,
                   WidthSeverity=1,
                   AngleSeverity=math.pi / 9,
                   TauSeverity=0.2,
                   EtaSeverity=10):
        self.dim = dim
        self.PeakNumber = PeakNumber
        self.ChangeFrequency = ChangeFrequency
        self.ShiftSeverity = ShiftSeverity
        self.EnvironmentNumber = EnvironmentNumber
        self.HeightSeverity = HeightSeverity
        self.WidthSeverity = WidthSeverity
        self.AngleSeverity = AngleSeverity
        self.TauSeverity = TauSeverity
        self.EtaSeverity = EtaSeverity
        self.maxfes = self.ChangeFrequency * self.EnvironmentNumber
        self.Environmentcounter = 0
        self.RecentChange = 0
        self.Ebbc = np.full(self.EnvironmentNumber, np.nan)
        self.CurrentError = np.full(self.maxfes + 1, np.nan)
        self.CurrentPerformance = np.full(self.maxfes + 1, np.nan)
        self.lb = -50
        self.ub = 50
        self.MinHeight = 30
        self.MaxHeight = 70
        self.MinWidth = 1
        self.MaxWidth = 12
        self.MinAngle = -math.pi
        self.MaxAngle = math.pi
        self.MinTau = 0.1
        self.MaxTau = 1
        self.MinEta = 0
        self.MaxEta = 50
        self.OptimumValue = np.full(self.EnvironmentNumber, np.nan)
        self.OptimumID = np.full(self.EnvironmentNumber, np.nan)
        self.PeaksHeight = np.full((self.EnvironmentNumber, self.PeakNumber), np.nan)
        self.PeaksPosition = np.full((self.PeakNumber, self.dim, self.EnvironmentNumber), np.nan)
        self.PeaksWidth = np.full((self.PeakNumber, self.dim, self.EnvironmentNumber), np.nan)
        self.PeaksPosition[:, :, 0] = self.lb + (self.ub - self.lb) * np.random.rand(self.PeakNumber, self.dim)
        self.PeaksHeight[0, :] = self.MinHeight + (self.MaxHeight - self.MinHeight) * np.random.rand(self.PeakNumber)
        self.PeaksWidth[:, :, 0] = self.MinWidth + (self.MaxWidth - self.MinWidth) * np.random.rand(self.PeakNumber, self.dim)
        self.OptimumID[0] = np.argmax(self.PeaksHeight[0, :])
        self.OptimumValue[0] = np.max(self.PeaksHeight[0, :])
        self.InitialRotationMatrix = np.full((self.dim, self.dim, self.PeakNumber), np.nan)
        for i in range(self.PeakNumber):
            q, _ = np.linalg.qr(np.random.rand(self.dim, self.dim))
            self.InitialRotationMatrix[:, :, i] = q
        self.RotationMatrix = [copy.deepcopy(self.InitialRotationMatrix) for _ in range(self.EnvironmentNumber)]
        self.PeaksAngle = np.full((self.EnvironmentNumber, self.PeakNumber), np.nan)
        self.tau = np.full((self.EnvironmentNumber, self.PeakNumber), np.nan)
        self.eta = np.full((self.PeakNumber, 4, self.EnvironmentNumber), np.nan)
        self.PeaksAngle[0, :] = self.MinAngle + (self.MaxAngle - self.MinAngle) * np.random.rand(self.PeakNumber)
        self.tau[0, :] = self.MinTau + (self.MaxTau - self.MinTau) * np.random.rand(self.PeakNumber)
        self.eta[:, :, 0] = self.MinEta + (self.MaxEta - self.MinEta) * np.random.rand(self.PeakNumber, 4)
        for i in range(1, self.EnvironmentNumber):
            ShiftOffset = np.random.randn(self.PeakNumber, self.dim)
            norms = np.linalg.norm(ShiftOffset, axis=1, keepdims=True)
            norms[norms == 0] = 1e-8
            Shift = ShiftOffset / norms * self.ShiftSeverity
            PeaksPosition = self.PeaksPosition[:, :, i - 1] + Shift
            PeaksWidth = self.PeaksWidth[:, :, i - 1] + np.random.randn(self.PeakNumber, self.dim) * self.WidthSeverity
            PeaksHeight = self.PeaksHeight[i - 1, :] + np.random.randn(self.PeakNumber) * self.HeightSeverity
            PeaksAngle = self.PeaksAngle[i - 1, :] + np.random.randn(self.PeakNumber) * self.AngleSeverity
            PeaksTau = self.tau[i - 1, :] + np.random.randn(self.PeakNumber) * self.TauSeverity
            PeaksEta = self.eta[:, :, i-1] + np.random.randn(self.PeakNumber, 4) * self.EtaSeverity
            tmp = PeaksAngle > self.MaxAngle
            PeaksAngle[tmp] = 2 * self.MaxAngle - PeaksAngle[tmp]
            tmp = PeaksAngle < self.MinAngle
            PeaksAngle[tmp] = 2 * self.MinAngle - PeaksAngle[tmp]
            tmp = PeaksTau > self.MaxTau
            PeaksTau[tmp] = 2 * self.MaxTau - PeaksTau[tmp]
            tmp = PeaksTau < self.MinTau
            PeaksTau[tmp] = 2 * self.MinTau - PeaksTau[tmp]
            tmp = PeaksEta > self.MaxEta
            PeaksEta[tmp] = 2 * self.MaxEta - PeaksEta[tmp]
            tmp = PeaksEta < self.MinEta
            PeaksEta[tmp] = 2 * self.MinEta - PeaksEta[tmp]
            tmp = PeaksPosition > self.ub
            PeaksPosition[tmp] = 2 * self.ub - PeaksPosition[tmp]
            tmp = PeaksPosition < self.lb
            PeaksPosition[tmp] = 2 * self.lb - PeaksPosition[tmp]
            tmp = PeaksHeight > self.MaxHeight
            PeaksHeight[tmp] = 2 * self.MaxHeight - PeaksHeight[tmp]
            tmp = PeaksHeight < self.MinHeight
            PeaksHeight[tmp] = 2 * self.MinHeight - PeaksHeight[tmp]
            tmp = PeaksWidth > self.MaxWidth
            PeaksWidth[tmp] = 2 * self.MaxWidth - PeaksWidth[tmp]
            tmp = PeaksWidth < self.MinWidth
            PeaksWidth[tmp] = 2 * self.MinWidth - PeaksWidth[tmp]
            self.PeaksPosition[:, :, i] = PeaksPosition
            self.PeaksWidth[:, :, i] = PeaksWidth
            self.PeaksHeight[i, :] = PeaksHeight
            self.PeaksAngle[i, :] = PeaksAngle
            self.tau[i, :] = PeaksTau
            self.eta[:, :, i] = PeaksEta
            for j in range(self.PeakNumber):
                self.RotationMatrix[i][:, :, j] = self.InitialRotationMatrix[:, :, j] @ self.Rotation(self.PeaksAngle[i, j])
            self.OptimumID[i] = np.argmax(PeaksHeight)
            self.OptimumValue[i] = np.max(PeaksHeight)

    def Rotation(self, teta):
        counter = 0
        PageNumber = int(self.dim * ((self.dim - 1) / 2))
        X = np.full((self.dim, self.dim, PageNumber), np.nan)
        for i in range(self.dim):
            for j in range(i + 1, self.dim):
                TmpMatrix = np.eye(self.dim)
                TmpMatrix[i, i] = np.cos(teta)
                TmpMatrix[j, j] = np.cos(teta)
                TmpMatrix[i, j] = np.sin(teta)
                TmpMatrix[j, i] = -np.sin(teta)
                X[:, :, counter] = TmpMatrix
                counter += 1
        output = np.eye(self.dim)
        tmp = np.random.permutation(PageNumber)
        for i in range(PageNumber):
            output = output @ X[:, :, tmp[i]]
        return output

    def fitness(self, X):
        x = X.T
        f = np.full(self.PeakNumber, np.nan)
        for k in range(self.PeakNumber):
            a_X = (x - self.PeaksPosition[k, :, self.Environmentcounter].T).T @ self.RotationMatrix[self.Environmentcounter][:, :, k].T
            b_X = self.RotationMatrix[self.Environmentcounter][:, :, k] @ (x - self.PeaksPosition[k, :, self.Environmentcounter].T)
            tau = self.tau[self.Environmentcounter, k]
            eta = self.eta[k, :, self.Environmentcounter]
            a = self.Transform(a_X, tau, eta)
            b = self.Transform(b_X, tau, eta)
            f[k] = self.PeaksHeight[self.Environmentcounter, k] - np.sqrt(a @ np.diag(self.PeaksWidth[k, :, self.Environmentcounter]) @ b)
        return np.max(f)

    def Transform(self, X, tau, eta):
        Y = X.copy()
        tmp = X > 0
        Y[tmp] = np.log(X[tmp])
        Y[tmp] = np.exp(Y[tmp] + tau * (np.sin(eta[0] * Y[tmp]) + np.sin(eta[1] * Y[tmp])))
        tmp = X < 0
        Y[tmp] = np.log(-X[tmp])
        Y[tmp] = -np.exp(Y[tmp] + tau * (np.sin(eta[2] * Y[tmp]) + np.sin(eta[3] * Y[tmp])))
        return Y

    def eval(self, X):
        SolutionNumber = 1 if X.ndim == 1 else X.shape[0]
        result = np.full(SolutionNumber, np.nan)
        for j in range(SolutionNumber):
            if self.fes >= self.maxfes or self.RecentChange:
                return np.full(SolutionNumber, -np.inf)
            result[j] = self.fitness(X[j])
            self.fes += 1
            SolutionError = self.OptimumValue[self.Environmentcounter] - result[j]
            if self.fes % self.ChangeFrequency != 1:
                if self.CurrentError[self.fes - 1] < SolutionError:
                    self.CurrentError[self.fes] = self.CurrentError[self.fes - 1]
                    self.CurrentPerformance[self.fes] = self.CurrentPerformance[self.fes - 1]
                else:
                    self.CurrentError[self.fes] = SolutionError
                    self.CurrentPerformance[self.fes] = result[j]
            else:
                self.CurrentError[self.fes] = SolutionError
                self.CurrentPerformance[self.fes] = result[j]
            if self.fes % self.ChangeFrequency == self.ChangeFrequency - 1:
                self.Ebbc[self.Environmentcounter] = self.CurrentError[self.fes]
            if self.fes % self.ChangeFrequency == 0 and self.fes < self.maxfes:
                self.Environmentcounter += 1
                self.RecentChange = 1
        return result

    def reset_RecentChange(self):
        self.RecentChange = 0


