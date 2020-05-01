from logging import getLogger
import numpy as np
import matplotlib.pyplot as plt

logger = getLogger(__name__)

class Visualize:
    def __init__(self):
        self.fig_realtime = plt.figure()
        self.ax_loss_generator = self.fig_realtime.add_subplot(231)
        self.ax_loss_discriminator = self.fig_realtime.add_subplot(232)
        self.ax_accuracy = self.fig_realtime.add_subplot(233)
        self.ax_original_points = self.fig_realtime.add_subplot(234)
        self.ax_generated_points = self.fig_realtime.add_subplot(235)
        self.label = ['Loss without regularization', 'Loss with regularization']
    
    def plot_realtime(self, losses_generator, losses_discriminator, accuracies, x_train, y_train, x_pred, y_pred):
        epoch = [i for i in range(1, len(losses_discriminator[0])+1)]
        self.ax_loss_generator.cla()
        for i, loss in enumerate(losses_generator):
            self.ax_loss_generator.plot(epoch, loss, label=self.label[i])
        self.ax_loss_generator.set_xlabel('Epoch')
        self.ax_loss_generator.set_ylabel('Loss')
        self.ax_loss_generator.set_title('Loss of generator')
        self.ax_loss_generator.legend()

        self.ax_loss_discriminator.cla()
        for i, loss in enumerate(losses_discriminator):
            self.ax_loss_discriminator.plot(epoch, loss, label=self.label[i])
        self.ax_loss_discriminator.set_xlabel('Epoch')
        self.ax_loss_discriminator.set_ylabel('Loss')
        self.ax_loss_discriminator.set_title('Loss of discriminator')
        self.ax_loss_discriminator.legend()

        self.ax_accuracy.cla()
        self.ax_accuracy.plot(epoch, accuracies)
        self.ax_accuracy.set_xlabel('Epoch')
        self.ax_accuracy.set_ylabel('Accuracy')
        self.ax_accuracy.set_title('Accuracy')

        self.ax_original_points.cla()
        self.ax_original_points.scatter(x_train[np.where(np.argmax(y_train, 1)==0)[0]][:,0], x_train[np.where(np.argmax(y_train, 1)==0)[0]][:,1], s=10, c='#ff0000', label='1')
        self.ax_original_points.scatter(x_train[np.where(np.argmax(y_train, 1)==1)[0]][:,0], x_train[np.where(np.argmax(y_train, 1)==1)[0]][:,1], s=10, c='#00ff00', label='2')
        self.ax_original_points.scatter(x_train[np.where(np.argmax(y_train, 1)==2)[0]][:,0], x_train[np.where(np.argmax(y_train, 1)==2)[0]][:,1], s=10, c='#0000ff', label='3')
        self.ax_original_points.set_xlabel(r'$x_1$')
        self.ax_original_points.set_ylabel(r'$x_2$')
        self.ax_original_points.set_title('Training data')
        self.ax_original_points.legend()

        self.ax_generated_points.cla()
        self.ax_generated_points.scatter(x_pred[np.where(np.argmax(y_pred, 1)==0)[0]][:,0], x_pred[np.where(np.argmax(y_pred, 1)==0)[0]][:,1], s=10, c='#ff0000', label='1')
        self.ax_generated_points.scatter(x_pred[np.where(np.argmax(y_pred, 1)==1)[0]][:,0], x_pred[np.where(np.argmax(y_pred, 1)==1)[0]][:,1], s=10, c='#00ff00', label='2')
        self.ax_generated_points.scatter(x_pred[np.where(np.argmax(y_pred, 1)==2)[0]][:,0], x_pred[np.where(np.argmax(y_pred, 1)==2)[0]][:,1], s=10, c='#0000ff', label='3')
        self.ax_generated_points.set_xlabel(r'$x_1$')
        self.ax_generated_points.set_ylabel(r'$x_2$')
        self.ax_generated_points.set_title('Generated points')
        self.ax_generated_points.legend()

        self.fig_realtime.tight_layout()
        self.fig_realtime.suptitle('Epoch: {}'.format(epoch[-1]))
        self.fig_realtime.subplots_adjust(top=0.92)
        plt.draw()
        plt.pause(0.00000000001)
    
    def save_plot_loss(self, losses, xlabel=None, ylabel=None, title=None, save_file=None):
        plt.clf()
        epoch = [i for i in range(len(losses[0]))]
        if xlabel is not None: plt.xlabel(xlabel)
        if ylabel is not None: plt.ylabel(ylabel)
        if title is not None: plt.title(title)
        for i, loss in enumerate(losses):
            plt.plot(epoch, loss, label=self.label[i])
        plt.legend()
        if save_file is None:
            plt.show()
        else:
            logger.debug("save plot of loss to {}".format(save_file))
            plt.savefig(save_file)
    
    def save_plot_accuracy(self, accuracies, xlabel=None, ylabel=None, title=None, save_file=None):
        plt.clf()
        epoch = [i for i in range(len(accuracies))]
        if xlabel is not None: plt.xlabel(xlabel)
        if ylabel is not None: plt.ylabel(ylabel)
        if title is not None: plt.title(title)
        plt.plot(epoch, accuracies)
        plt.legend()
        if save_file is None:
            plt.show()
        else:
            logger.debug("save plot of accuracy to {}".format(save_file))
            plt.savefig(save_file)
    
    def save_plot_params(self, t, params, save_file=None):
        plt.clf()
        alpha_x, beta_x, gamma_x, A_x, alpha_y, beta_y, gamma_y, A_y = params
        fig_alpha_x = plt.figure()
        ax_alpha_x = fig_alpha_x.add_subplot(111)
        ax_alpha_x.set_xlabel('t')
        ax_alpha_x.set_ylabel(r'$\alpha_x(t)$')
        ax_alpha_x.set_title(r'$\alpha_x(t)$')

        fig_beta_x = plt.figure()
        ax_beta_x = fig_beta_x.add_subplot(111)
        ax_beta_x.set_xlabel('t')
        ax_beta_x.set_ylabel(r'$\beta_x(t)$')
        ax_beta_x.set_title(r'$\beta_x(t)$')

        fig_gamma_x = plt.figure()
        ax_gamma_x = fig_gamma_x.add_subplot(111)
        ax_gamma_x.set_xlabel('t')
        ax_gamma_x.set_ylabel(r'$\gamma_x(t)$')
        ax_gamma_x.set_title(r'$\gamma_x(t)$')

        fig_A_x = plt.figure()
        ax_A_x = fig_A_x.add_subplot(111)
        ax_A_x.set_xlabel('t')
        ax_A_x.set_ylabel(r'$A_x(t)$')
        ax_A_x.set_title(r'$A_x(t)$')

        fig_alpha_y = plt.figure()
        ax_alpha_y = fig_alpha_y.add_subplot(111)
        ax_alpha_y.set_xlabel('t')
        ax_alpha_y.set_ylabel(r'$\alpha_y(t)$')
        ax_alpha_y.set_title(r'$\alpha_y(t)$')

        fig_beta_y = plt.figure()
        ax_beta_y = fig_beta_y.add_subplot(111)
        ax_beta_y.set_xlabel('t')
        ax_beta_y.set_ylabel(r'$\beta_y(t)$')
        ax_beta_y.set_title(r'\beta_y(t)$')

        fig_gamma_y = plt.figure()
        ax_gamma_y = fig_gamma_y.add_subplot(111)
        ax_gamma_y.set_xlabel('t')
        ax_gamma_y.set_ylabel(r'$\gamma_y(t)$')
        ax_gamma_y.set_title(r'$\gamma_y(t)$')

        fig_A_y = plt.figure()
        ax_A_y = fig_A_y.add_subplot(111)
        ax_A_y.set_xlabel('t')
        ax_A_y.set_ylabel(r'$A_y(t)$')
        ax_A_y.set_title(r'$A_y(t)$')

        fig_params = plt.figure()
        ax_params = fig_params.add_subplot(111)
        ax_params.set_xlabel('t')
        ax_params.set_title('Parameter')

        for i in range(len(alpha_x[0])):
            ax_alpha_x.plot(t, alpha_x[:, i], label=r'$\alpha_{}^x(t)$'.format(i))
            ax_params.plot(t, alpha_x[:, i], label=r'$\alpha_{}^x(t)$'.format(i))
        for i in range(len(beta_x[0])):
            for j in range(len(beta_x[0,0])):
                ax_beta_x.plot(t, beta_x[:, i, j], label=r'$\beta_{}$$_{}^x(t)$'.format(i,j))
                ax_params.plot(t, beta_x[:, i, j], label=r'$\beta_{}$$_{}^x(t)$'.format(i,j))
        for i in range(len(gamma_x[0])):
            ax_gamma_x.plot(t, gamma_x[:, i], label=r'$\gamma_{}^x(t)$'.format(i))
            ax_params.plot(t, gamma_x[:, i], label=r'$\gamma_{}^x(t)$'.format(i))
        for i in range(len(A_x[0])):
            for j in range(len(A_x[0,0])):
                ax_A_x.plot(t, A_x[:, i, j], label=r'$A_{}$$_{}^x(t)$'.format(i, j))
                ax_params.plot(t, A_x[:, i, j], label=r'$A_{}$$_{}^x(t)$'.format(i, j))
        for i in range(len(alpha_y[0])):
            ax_alpha_y.plot(t, alpha_y[:, i], label=r'$\alpha_{}^y(t)$'.format(i))
            ax_params.plot(t, alpha_y[:, i], label=r'$\alpha_{}^y(t)$'.format(i))
        for i in range(len(beta_y[0])):
            for j in range(len(beta_y[0,0])):
                ax_beta_y.plot(t, beta_y[:, i, j], label=r'$\beta_{}$$_{}^y(t)$'.format(i,j))
                ax_params.plot(t, beta_y[:, i, j], label=r'$\beta_{}$$_{}^y(t)$'.format(i,j))
        for i in range(len(gamma_y[0])):
            ax_gamma_y.plot(t, gamma_y[:, i], label=r'$\gamma_{}^y(t)$'.format(i))
            ax_params.plot(t, gamma_y[:, i], label=r'$\gamma_{}^y(t)$'.format(i))
        for i in range(len(A_y[0])):
            for j in range(len(A_y[0,0])):
                ax_A_y.plot(t, A_y[:, i, j], label=r'$A_{}$$_{}^y(t)$'.format(i, j))
                ax_params.plot(t, A_y[:, i, j], label=r'$A_{}$$_{}^y(t)$'.format(i, j))
        
        ax_alpha_x.legend()
        ax_beta_x.legend()
        ax_gamma_x.legend()
        ax_A_x.legend()
        ax_alpha_y.legend()
        ax_beta_y.legend()
        ax_gamma_y.legend()
        ax_A_y.legend()

        if save_file is not None:
            logger.debug("save plot of parameter0 (alpha_x) to {}".format(save_file[0]))
            fig_alpha_x.savefig(save_file[0])
            logger.debug("save plot of parameter1 (beta_x) to {}".format(save_file[1]))
            fig_beta_x.savefig(save_file[1])
            logger.debug("save plot of parameter2 (gamma_x) to {}".format(save_file[2]))
            fig_gamma_x.savefig(save_file[2])
            logger.debug("save plot of parameter3 (A_x) to {}".format(save_file[3]))
            fig_A_x.savefig(save_file[3])
            logger.debug("save plot of parameter4 (alpha_y) to {}".format(save_file[4]))
            fig_alpha_y.savefig(save_file[4])
            logger.debug("save plot of parameter5 (beta_y) to {}".format(save_file[5]))
            fig_beta_y.savefig(save_file[5])
            logger.debug("save plot of parameter6 (gamma_y) to {}".format(save_file[6]))
            fig_gamma_y.savefig(save_file[6])
            logger.debug("save plot of parameter7 (A_y) to {}".format(save_file[7]))
            fig_A_y.savefig(save_file[7])
            logger.debug("save plot of parameters to {}".format(save_file[8]))
            fig_params.savefig(save_file[8])