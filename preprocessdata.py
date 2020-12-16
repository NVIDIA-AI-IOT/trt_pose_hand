import math
import pickle
import cv2


class preprocessdata:
    
    def __init__(self, topology, num_parts):
        self.joints = []
        self.dist_bn_joints = []
        self.topology = topology
        self.num_parts = num_parts
        self.text = "no hand"
        self.num_frames = 4
        self.prev_queue = [ self.num_frames ]*self.num_frames
        
    def svm_accuracy(self, test_predicted, labels_test):
        """"
        This method calculates the accuracy of the model 
        Input: test_predicted - predicted test classes
               labels_test
        Output: accuracy - of the model 
        """
        predicted = []
        for i in range(len(labels_test)):
            if labels_test[i]==test_predicted[i]:
                predicted.append(0)
            else:
                predicted.append(1)
        accuracy = 1 - sum(predicted)/len(labels_test)
        return accuracy 
    def trainsvm(self, clf, train_data, test_data, labels_train, labels_test):
        """
        This method trains the different gestures 
        Input: clf - Sk-learn model pipeline to train, You can choose an SVM, linear regression, etc
                train_data - preprocessed training image data -in this case the distance between the joints
                test_data - preprocessed testing image data -in this case the distance between the joints
                labels_train - labels for training images 
                labels_test - labels for testing images 
        Output: trained model, predicted_test_classes
        """
        clf.fit(train_data,labels_train)
        predicted_test = clf.predict(test_data)
        return clf, predicted_test   
    #def loadsvmweights():
    
    def joints_inference(self, image, counts, objects, peaks): 
        """
        This method returns predicted joints from an image/frame
        Input: image, counts, objects, peaks
        Output: predicted joints
        """
        joints_t = []
        height = image.shape[0]
        width = image.shape[1]
        K = self.topology.shape[0]
        count = int(counts[0])
        for i in range(count):
            obj = objects[0][i]
            C = obj.shape[0]
            for j in range(C):
                k = int(obj[j])
                picked_peaks = peaks[0][j][k]
                joints_t.append([round(float(picked_peaks[1]) * width), round(float(picked_peaks[0]) * height)])
        joints_pt = joints_t[:self.num_parts]  
        rest_of_joints_t = joints_t[self.num_parts:]
        
        #when it does not predict a particular joint in the same association it will try to find it in a different association 
        for i in range(len(rest_of_joints_t)):
            l = i%self.num_parts
            if joints_pt[l] == [0,0]:
                joints_pt[l] = rest_of_joints_t[i]
                
        #if nothing is predicted 
        if count == 0:
            joints_pt = [[0,0]]*self.num_parts
        return joints_pt
    def find_distance(self, joints):
        """
        This method finds the distance between each joints 
        Input: a list that contains the [x,y] positions of the 21 joints 
        Output: a list that contains the distance between the joints 
        """
        joints_features = []
        for i in joints:
            for j in joints:
                dist_between_i_j = math.sqrt((i[0]-j[0])**2+(i[1]-j[1])**2)
                joints_features.append(dist_between_i_j)
        return joints_features
    def print_label(self, image, gesture_joints, gesture_type):
        """
        This method prints the gesture class detected. 
        Example. Incase of the cursor control application it shows if your gesture is a click or other type of gesture
        
        """
        font = cv2.FONT_HERSHEY_SIMPLEX 
        color = (255, 0, 0) 
        org = (50, 50)
        thickness = 2
        fontScale = 0.5
        no_frames = 4
        if self.prev_queue == [1]* self.num_frames:
            self.text = gesture_type[0]
        elif self.prev_queue == [2]* self.num_frames:
            self.text = gesture_type[1]
        elif self.prev_queue == [3]* self.num_frames:
            self.text = gesture_type[2]
        elif self.prev_queue == [4]* self.num_frames:
            self.text = gesture_type[3]
        elif self.prev_queue == [5]* self.num_frames:
            self.text = gesture_type[4]
        elif self.prev_queue == [6]* self.num_frames:
            self.text = gesture_type[5]
        elif self.prev_queue == [7]*self.num_frames:
            self.text = gesture_type[6]
        image = cv2.putText(image, self.text, org, font,  
                       fontScale, color, thickness, cv2.LINE_AA) 
        return image
