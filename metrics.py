import torch
import numpy as np

def get_metrics(metric_name, classes):

    if metric_name == "dice_score":
        dice = DiceScore(classes=classes)
        return dice
    

class ImageMetrics():


    def __init__(self, classes, eps = 0.0000001):

        self.classes = classes
        self.eps = eps

        self.global_intersection_sum = np.zeros((classes))
        self.global_prediction_sum  = np.zeros((classes))
        self.global_gt_sum  = np.zeros((classes))

        self.batch_dice = np.zeros((classes))
        self.batch_precision = np.zeros((classes))
        self.batch_recall = np.zeros((classes))
        

    def fit(self, y_pred, y_true):

        for class_index in range(self.classes):

            temp_y_pred = torch.zeros_like(y_pred)
            temp_y_true = torch.zeros_like(y_true)

            temp_y_pred[y_pred == class_index] = 1
            temp_y_true[y_true == class_index] = 1

            batch_intersection_sum = torch.sum(temp_y_pred * temp_y_true)
            batch_prediction_sum = torch.sum(temp_y_pred)
            batch_gt_sum = torch.sum(temp_y_true)
            
            self.global_intersection_sum[class_index] += batch_intersection_sum
            self.global_prediction_sum[class_index] += batch_prediction_sum
            self.global_gt_sum[class_index] += batch_gt_sum

            # Score Calculation
            self.batch_dice[class_index] = 2*batch_intersection_sum/(batch_prediction_sum + batch_gt_sum + self.eps)
            self.batch_precision[class_index] = 2*batch_intersection_sum/(batch_prediction_sum + self.eps)
            self.batch_recall[class_index] = 2*batch_prediction_sum/(batch_gt_sum + self.eps)
    
    
    def get_global_dice(self):
        global_dice = 2*(self.global_intersection_sum)/(self.global_prediction_sum + self.global_gt_sum + self.eps)
        return global_dice
    
    def get_global_precision(self):
        global_precision = (self.global_intersection_sum)/(self.global_prediction_sum + self.eps)
        return global_precision

    def get_global_recall(self):
        global_recall = (self.global_intersection_sum)/(self.global_gt_sum + self.eps)
        return global_recall
    
    def get_batch_dice(self):
        return self.batch_dice
    
    def get_batch_precision(self):
        return self.batch_precision
    
    def get_batch_recall(self):
        return self.batch_recall
    


    







class DiceScore():

    def __init__(self,classes):
        self.classes = classes
        self.intersection = np.zeros((classes))
        self.union  = np.zeros((classes))

    def calculate_batch_dice(self, y_pred, y_true):

        class_wise_dice_list = []


        for class_index in range(self.classes):

            temp_y_pred = torch.zeros_like(y_pred)
            temp_y_true = torch.zeros_like(y_true)

            temp_y_pred[y_pred == class_index] = 1
            temp_y_true[y_true == class_index] = 1

            temp_intersection = torch.sum(temp_y_pred * temp_y_true)
            temp_union = torch.sum(temp_y_pred + temp_y_true)
            
            self.intersection[class_index] += temp_intersection
            self.union[class_index] += temp_union
            
            dice = (2*temp_intersection +  0.0000001)/(temp_union + 0.0000001)
            class_wise_dice_list.append(dice)

        return class_wise_dice_list
    
    def calculate_global_dice(self):

        classwise_global_dice = []
        
        for class_index in range(self.classes):
            global_dice = (2*self.intersection[class_index] +  0.0000001)/(self.union[class_index] + 0.0000001)
            classwise_global_dice.append(global_dice)

        return classwise_global_dice
    
    def reset(self):
        self.intersection = np.zeros((self.classes))
        self.union  = np.zeros((self.classes))
    

        

        

        
