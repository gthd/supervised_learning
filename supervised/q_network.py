import vrep
import time
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import pickle
import numpy as np
import buffer
import itertools
from collections import namedtuple
import os
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import robot

class Q_Network(nn.Module):
    def __init__(self):
        super(Q_Network, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.conv0 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.bn0 = nn.BatchNorm2d(32)
        self.conv1 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.fc_model_0 = nn.Linear(in_features=4*4*64, out_features=256)
        self.fc_model_1 = nn.Linear(in_features=256, out_features=256)
        self.fc_model_2 = nn.Linear(in_features=256, out_features=64)

        self.fc0_1 = nn.Linear(3, 16)
        self.fc0_2 = nn.Linear(16, 32)
        self.fc0_3 = nn.Linear(32, 64)

        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 16)
        self.fc5 = nn.Linear(16, 1)

        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, x, action_vector):

        x = x.to(self.device)

        x = self.bn0(self.relu(self.conv0(x)))
        x = self.bn1(self.relu(self.conv1(x)))
        x = self.bn2(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc_model_0(x))
        x = self.relu(self.fc_model_1(x))
        x = self.relu(self.fc_model_2(x))

        action_vector = action_vector.to(self.device)

        action_vector = self.relu(self.fc0_1(action_vector))
        action_vector = self.relu(self.fc0_2(action_vector))
        action_vector = self.relu(self.fc0_3(action_vector))

        new_vector = action_vector + x

        new_vector = self.relu(self.fc1(new_vector))
        new_vector = self.relu(self.fc2(new_vector))
        new_vector = self.relu(self.fc3(new_vector))
        new_vector = self.relu(self.fc4(new_vector))
        probability = self.sigmoid(self.fc5(new_vector))
        return probability

class ParseData:
    def __init__(self):
        self.robot = robot
        self.criterion = nn.BCELoss()
        torch.cuda.empty_cache()
        self.q_network = Q_Network().cuda()
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=1e-5)
        print(sum(p.numel() for p in self.q_network.parameters() if p.requires_grad))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.valid_loss_list_epoch = []
        self.valid_acc_list_epoch = []

    def train(self, batch_size=2, epochs=30): #should get the number of samples
        dirs = os.listdir('/home/george/Desktop/Github/supervised_learning/Datasets/train_data/')
        number_of_files = 35
        the_loss = []
        the_acc = []
        for epoch in range(epochs):
            number_of_samples = 0
            num_file = 0
            jj = 0
            loss_list_epoch = []
            acc_list_epoch = []
            for file in dirs:
                acc_list = []
                loss_list = []
                num_file += 1
                pkl_file = open('/home/george/Desktop/Github/supervised_learning/Datasets/train_data/'+file, 'rb')
                objects = []
                while True:
                    try:
                        objects.append(pickle.load(pkl_file))
                    except EOFError:
                        break
                pkl_file.close()
                list_grasp_pos_x = []
                list_grasp_pos_y = []
                list_grasp_orientation = []
                list_image = []
                list_label = []
                new_objects = []
                for obj in objects:
                    if obj.image.shape[0] == 32:
                        new_objects.append(obj)
                N = len(new_objects)
                number_of_samples += N
                # print(file)
                # print(N)
                for i in range(N):
                    jj += 1
                    if i % batch_size == 0 and i != 0:
                        img_state_batch = torch.cat(list_image)
                        img_state_batch = img_state_batch.clone().detach().float()
                        grasp_pos_x_batch = torch.cat(list_grasp_pos_x).float()
                        grasp_pos_y_batch = torch.cat(list_grasp_pos_y).float()
                        grasp_orientation_batch = torch.cat(list_grasp_orientation).float()
                        label_batch = torch.cat(list_label).float()
                        label_batch = label_batch.view(-1, 1).float().to(self.device)
                        grasp_orientation_batch = grasp_orientation_batch.view(-1,1)
                        grasp_pos_x_batch = grasp_pos_x_batch.view(-1,1)
                        grasp_pos_y_batch = grasp_pos_y_batch.view(-1,1)
                        img_state_batch = img_state_batch.view(-1,3, 32, 32)
                        action_vector = torch.cat((grasp_pos_x_batch, grasp_pos_y_batch, grasp_orientation_batch), dim=1)

                        probabilities = self.q_network(img_state_batch, action_vector)
                        loss = self.criterion(probabilities, label_batch)
                        # print(probabilities)
                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()
                        total = action_vector.shape[0]
                        a = []
                        for km in probabilities:
                            if km > 0.5:
                                a.append(1)
                            else:
                                a.append(0)
                        a = torch.tensor(a).float()

                        h = []
                        for kl in label_batch:
                            h.append(kl[0])
                        h = torch.tensor(h)

                        correct = (h == a).sum().item()
                        acc_list.append((float(correct)/total)*100)
                        loss_list.append(loss.item())

                        list_grasp_pos_x = []
                        list_grasp_pos_y = []
                        list_grasp_orientation = []
                        list_image = []
                        list_label = []
                    else:
                        list_grasp_pos_x.append(torch.tensor(new_objects[i].grasp_pos_x).view(1,))
                        list_grasp_pos_y.append(torch.tensor(new_objects[i].grasp_pos_y).view(1,))
                        list_grasp_orientation.append(torch.tensor(new_objects[i].grasp_orientation).view(1,))
                        list_image.append(torch.from_numpy(new_objects[i].image))
                        list_label.append(torch.tensor(new_objects[i].label).view(1,))

                print('Epoch [{}/{}], File[{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%').format(epoch+1, epochs, num_file, number_of_files, sum(loss_list)/len(loss_list), sum(acc_list)/len(acc_list))
                loss_list_epoch.append(sum(loss_list)/len(loss_list))
                acc_list_epoch.append(sum(acc_list)/len(acc_list))

            print('Over Epoch, Epoch [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%').format(epoch+1, epochs, sum(loss_list_epoch)/len(loss_list_epoch), sum(acc_list_epoch)/len(acc_list_epoch))
            print('number of training samples', number_of_samples)
            torch.save(self.q_network.state_dict(), "/homes/gt4118/Desktop/supervised_learning/weights_classification.pth")
            self.validate()

            for l in loss_list_epoch:
                the_loss.append(l)
            for a in acc_list_epoch:
                the_acc.append(a)
            self.plot(the_loss, the_acc)

    def plot(self, train_loss, train_accuracy):
        train_ind = np.arange(len(train_loss))
        valid_ind = np.arange(len(self.valid_loss_list_epoch))

        plt.plot(train_ind, train_loss)
        plt.xlabel('iteration per file')
        plt.ylabel('Loss')
        plt.suptitle('Training Loss Plot')
        plt.show()
        plt.pause(3)
        plt.savefig('/homes/gt4118/Desktop/supervised_learning/train_loss_plot.jpg')
        plt.cla()

        plt.plot(valid_ind, self.valid_loss_list_epoch)
        plt.xlabel('iteration per file')
        plt.ylabel('Loss')
        plt.suptitle('Validation Loss Plot')
        plt.show()
        plt.pause(3)
        plt.savefig('/home/gt4118/Desktop/supervised_learning/valid_loss_plot.jpg')
        plt.cla()

        plt.plot(train_ind, train_accuracy, '-b')
        plt.xlabel('iteration per file')
        plt.ylabel('Accuracy')
        plt.suptitle('Training Accuracy Plot')
        plt.show()
        plt.pause(3)
        plt.savefig('/home/gt4118/Desktop/supervised_learning/train_acc_plot.jpg')
        plt.cla()

        plt.plot(valid_ind, self.valid_acc_list_epoch, '-b')
        plt.xlabel('iteration per file')
        plt.ylabel('Accuracy')
        plt.suptitle('Validation Accuracy Plot')
        plt.show()
        plt.pause(3)
        plt.savefig('/home/gt4118/Desktop/supervised_learning/valid_acc_plot.jpg')
        plt.cla()

    def validate(self):
        dirs = os.listdir('/homes/gt4118/Desktop/supervised_learning/Datasets/validation_data/')
        batch_size = 25
        number_of_files = 9
        self.q_network.load_state_dict(torch.load("/homes/gt4118/Desktop/supervised_learning/weights_classification.pth"))
        self.q_network.eval()
        correct = 0
        total = 0
        number_of_valid_samples = 0
        num_file = 0
        valid_loss_epoch = []
        valid_acc_epoch = []
        with torch.no_grad():
            for file in dirs:
                num_file += 1
                valid_loss_list = []
                valid_acc_list = []
                pkl_file = open('/homes/gt4118/Desktop/supervised_learning/Datasets/validation_data/'+file, 'rb')
                objects = []
                while True:
                    try:
                        objects.append(pickle.load(pkl_file))
                    except EOFError:
                        break
                pkl_file.close()
                list_grasp_pos_x = []
                list_grasp_pos_y = []
                list_grasp_orientation = []
                list_image = []
                list_label = []
                new_objects = []
                print('len objects', len(objects))
                for obj in objects:
                    if obj.image.shape[0] == 32:
                        new_objects.append(obj)
                objects = new_objects
                # for obj in new_objects:
                #     if obj.label == 1:
                #         objects.append(obj)
                N = len(objects)
                number_of_valid_samples += N
                for i in range(N):
                    if i%batch_size == 0 and i != 0:
                        img_state_batch = torch.cat(list_image)
                        img_state_batch = img_state_batch.clone().detach().float()
                        grasp_pos_x_batch = torch.cat(list_grasp_pos_x).float()
                        grasp_pos_y_batch = torch.cat(list_grasp_pos_y).float()
                        grasp_orientation_batch = torch.cat(list_grasp_orientation).float()
                        label_batch = torch.cat(list_label).float()
                        label_batch = label_batch.view(-1, 1).float().to(self.device)
                        grasp_orientation_batch = grasp_orientation_batch.view(-1,1)
                        grasp_pos_x_batch = grasp_pos_x_batch.view(-1,1)
                        grasp_pos_y_batch = grasp_pos_y_batch.view(-1,1)
                        img_state_batch = img_state_batch.view(-1,3, 32, 32)
                        action_vector = torch.cat((grasp_pos_x_batch, grasp_pos_y_batch, grasp_orientation_batch), dim=1)
                        probabilities = self.q_network(img_state_batch, action_vector)
                        loss = self.criterion(probabilities, label_batch)
                        valid_loss_list.append(loss.item())
                        total = action_vector.shape[0]
                        a = []
                        for km in probabilities:
                            if km > 0.5:
                                a.append(1)
                            else:
                                a.append(0)
                        a = torch.tensor(a).float()

                        h = []
                        for kl in label_batch:
                            h.append(kl[0])
                        h = torch.tensor(h)

                        correct = (h == a).sum().item()
                        valid_acc_list.append((float(correct)/total)*100)
                        list_grasp_pos_x = []
                        list_grasp_pos_y = []
                        list_grasp_orientation = []
                        list_image = []
                        list_label = []

                    else:
                        list_grasp_pos_x.append(torch.tensor(objects[i].grasp_pos_x).view(1,))
                        list_grasp_pos_y.append(torch.tensor(objects[i].grasp_pos_y).view(1,))
                        list_grasp_orientation.append(torch.tensor(objects[i].grasp_orientation).view(1,))
                        list_image.append(torch.from_numpy(objects[i].image))
                        list_label.append(torch.tensor(objects[i].label).view(1,))

                print('File[{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%').format(num_file, number_of_files, sum(valid_loss_list)/len(valid_loss_list), sum(valid_acc_list)/len(valid_acc_list))
                valid_loss_epoch.append(sum(valid_loss_list)/len(valid_loss_list))
                valid_acc_epoch.append(sum(valid_acc_list)/len(valid_acc_list))

            print('Overall Loss: {:.4f}, Accuracy: {:.2f}%').format(sum(valid_loss_epoch)/len(valid_loss_epoch), sum(valid_acc_epoch)/len(valid_acc_epoch))
            print('number of validation samples', number_of_valid_samples)
            for l in valid_loss_epoch:
                self.valid_loss_list_epoch.append(l)
            for a in valid_acc_epoch:
                self.valid_acc_list_epoch.append(a)
            time.sleep(4)

my_data = ParseData()
my_data.train()
