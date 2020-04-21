import pickle
import os
import robot
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision.models as models
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('agg')

class QNetwork(nn.Module):
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        super(QNetwork, self).__init__()
        self.model_ft = models.resnet34(pretrained=True)
        self.model_fc = nn.Linear(1000, 3)
        self.relu = nn.ReLU()

    def forward(self, image):

        image = image.to(self.device)
        image = self.model_ft(image)
        image = self.relu(image)
        out = self.model_fc(image)

        return out

class Regression:
    def __init__(self):
        self.robot = robot
        self.criterion = nn.MSELoss()
        torch.cuda.empty_cache()
        self.q_network = QNetwork().cuda()
        # self.q_network.load_state_dict(torch.load("/homes/gt4118/Desktop/Robot_Learning/
        # weights.pth"))
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=1e-6)
        # print(sum(p.numel() for p in self.q_network.parameters() if p.requires_grad))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def train(self, batch_size=10, epochs=30): #should get the number of samples
        dirs = os.listdir('/home/george/Desktop/Github/supervised_learning/Datasets/train_data/')
        number_of_files = 35
        loss_list_epoch = []
        for epoch in range(epochs):
            number_of_samples = 0
            num_file = 0
            for file in dirs:
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
                    if obj.image.shape[0] == 224 and obj.label == 1:
                        new_objects.append(obj)
                number_of_objects = len(new_objects)
                number_of_samples += number_of_objects
                for i in range(number_of_objects):
                    if i % batch_size == 0 and i != 0:
                        img_state_batch = torch.cat(list_image)
                        img_state_batch = img_state_batch.clone().detach().float()
                        grasp_pos_x_batch = torch.cat(list_grasp_pos_x).float()
                        grasp_pos_y_batch = torch.cat(list_grasp_pos_y).float()
                        grasp_orientation_batch = torch.cat(list_grasp_orientation).float()
                        label_batch = torch.cat(list_label).float()
                        label_batch = label_batch.view(-1, 1).float().to(self.device)
                        grasp_orientation_batch = grasp_orientation_batch.view(-1, 1)
                        grasp_pos_x_batch = grasp_pos_x_batch.view(-1, 1)
                        grasp_pos_y_batch = grasp_pos_y_batch.view(-1, 1)
                        img_state_batch = img_state_batch.view(-1, 3, 224, 224)
                        action_vector = torch.cat((grasp_pos_x_batch, grasp_pos_y_batch, \
                            grasp_orientation_batch), dim=1)

                        predicted_action = self.q_network(img_state_batch)
                        loss = self.criterion(predicted_action, action_vector.to(self.device))
                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()
                        loss_list.append(loss.item())
                        list_grasp_pos_x = []
                        list_grasp_pos_y = []
                        list_grasp_orientation = []
                        list_image = []
                        list_label = []
                    else:
                        list_grasp_pos_x.append(torch.tensor(new_objects[i].grasp_pos_x).view(1,))
                        list_grasp_pos_y.append(torch.tensor(new_objects[i].\
                            grasp_pos_y).view(1,))
                        list_grasp_orientation.append(torch.tensor(new_objects[i].\
                            grasp_orientation).view(1,))
                        list_image.append(torch.from_numpy(new_objects[i].image))
                        list_label.append(torch.tensor(new_objects[i].\
                            label).view(1,))

                print('Epoch [{}/{}], File[{}/{}], Average Loss: {:.4f}').\
                    format(epoch+1, epochs, num_file, number_of_files, \
                    sum(loss_list)/len(loss_list))
                loss_list_epoch.append(sum(loss_list)/len(loss_list))

            print('Over Epoch, Epoch [{}/{}], Loss: {:.4f}').\
                format(epoch+1, epochs, sum(loss_list_epoch)/\
                len(loss_list_epoch))
            print('number of training samples', number_of_samples)
            self.plot(loss_list_epoch)
            # torch.save(self.q_network.state_dict(), "/home/george/Desktop/
            # Github/supervised_learning/weights_regression.pth")

    def plot(self, loss):
        ind = np.arange(len(loss))
        plt.plot(ind, loss, '-b')
        plt.title('Loss over Iteration per file')
        plt.show()
        plt.pause(3)
        plt.savefig(r"/home/george/Desktop/Github/supervised_learning" \
            r"/training.jpg")

    def validate(self):
        dirs = os.listdir(r"/home/george/Desktop/Github/supervised_learning/Datasets" \
            r"/train_data/")
        valid_loss_list_epoch = []
        batch_size = 10
        self.q_network.load_state_dict(torch.load("/home/george/Desktop/Github/supervised_learning/weights_regression.pth"))
        self.q_network.eval()
        number_of_valid_samples = 0
        num_file = 0
        with torch.no_grad():
            for file in dirs:
                num_file += 1
                valid_loss_list = []
                dist_x_list = []
                dist_y_list = []
                pkl_file = open(r"/home/george/Desktop" \
                    r"/Github/supervised_learning/Datasets/validation_data/" \
                    +file, 'rb')
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
                    if obj.image.shape[0] == 224:
                        new_objects.append(obj)
                objects = []
                for obj in new_objects:
                    if obj.label == 1:
                        objects.append(obj)
                number_of_objects = len(objects)
                number_of_valid_samples += number_of_objects
                for i in range(number_of_objects):
                    if i%batch_size == 0 and i != 0:
                        img_state_batch = torch.cat(list_image)
                        img_state_batch = img_state_batch.clone().detach().float()
                        grasp_pos_x_batch = torch.cat(list_grasp_pos_x).float()
                        grasp_pos_y_batch = torch.cat(list_grasp_pos_y).float()
                        grasp_orientation_batch = torch.cat(list_grasp_orientation).float()
                        label_batch = torch.cat(list_label).float()
                        label_batch = label_batch.view(-1, 1).float().to(self.device)
                        grasp_orientation_batch = grasp_orientation_batch.view(-1, 1)
                        grasp_pos_x_batch = grasp_pos_x_batch.view(-1, 1)
                        grasp_pos_y_batch = grasp_pos_y_batch.view(-1, 1)
                        img_state_batch = img_state_batch.view(-1, 3, 224, 224)
                        action_vector = torch.cat((grasp_pos_x_batch, grasp_pos_y_batch, \
                            grasp_orientation_batch), dim=1)
                        pred_action = self.q_network(img_state_batch)
                        loss = self.criterion(pred_action, action_vector.to(self.device))
                        for j in range(batch_size-1):
                            dist_x_list.append((pred_action[j][0] - action_vector[j][0]).item())
                            dist_y_list.append((pred_action[j][1] - action_vector[j][1]).item())

                        valid_loss_list.append(loss.item())

                        list_grasp_pos_x = []
                        list_grasp_pos_y = []
                        list_grasp_orientation = []
                        list_image = []
                        list_label = []

                    else:
                        list_grasp_pos_x.append(torch.tensor(objects[i].grasp_pos_x).view(1,))
                        list_grasp_pos_y.append(torch.tensor(objects[i].\
                            grasp_pos_y).view(1,))
                        list_grasp_orientation.append(torch.tensor(objects[i].\
                            grasp_orientation).view(1,))
                        list_image.append(torch.from_numpy(objects[i].image))
                        list_label.append(torch.tensor(objects[i].label).view(1,))

                print('File[{}/{}], Loss: {:.4f}').\
                    format(num_file, 44, sum(valid_loss_list)/\
                    len(valid_loss_list))
                print('avg distance x-axis: ', sum(np.abs(dist_x_list))/len(dist_x_list))
                print('avg distance y-axis: ', sum(np.abs(dist_y_list))/len(dist_y_list))
                valid_loss_list_epoch.append(sum(valid_loss_list)/len(valid_loss_list))

            print('Overall Loss: {:.4f}').format(sum(valid_loss_list_epoch)/\
                len(valid_loss_list_epoch))
rg = Regression()
rg.validate()
