import os
import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
from data_loader import DataLoader
from metrics import cal_all
from tqdm import tqdm

def print_gradient_statistics(model):
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            print(f"{name} - Grad Mean: {param.grad.mean()}, Grad Std: {param.grad.std()}")

class Trainer:
    def __init__(self, model, lr, device, num_epochs, batch_size, label_selet):
        self.model = model.to(device)
        # self.criterion = nn.MSELoss()
        self.criterion = nn.L1Loss()
        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-5)
        self.device = device
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.label_select = label_selet

    def train_model(self, train_data_path, val_data_path, test_data_path, save_folder):
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        data_loader = DataLoader()
        best_score = float('inf')
        best_model_path = None

        for epoch in range(self.num_epochs):
            if best_model_path is not None and os.path.exists(best_model_path):
                self.model.load_state_dict(torch.load(best_model_path))
                print(f"Loaded best model from {best_model_path} for epoch {epoch}")

            self.model.train()
            total_loss, total_y, total_pred = 0, [], []
            count = 0
            for batch_data in tqdm(data_loader.data_generator(train_data_path, self.batch_size, self.label_select)):
            # for batch_data in data_loader.data_generator(train_data_path, self.batch_size, self.label_select):
                time1 = time.time()
                batch_x, batch_masks, batch_token_type_ids, batch_refs_x, batch_y = batch_data
                batch_x, batch_masks, batch_token_type_ids, batch_y = map(lambda x: x.to(self.device), [batch_x, batch_masks, batch_token_type_ids, batch_y])
                batch_refs_x = batch_refs_x.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(batch_x, batch_masks, batch_token_type_ids, batch_refs_x)
                loss = self.criterion(outputs, batch_y)
                loss.backward()  # 计算梯度
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1)  # 裁剪梯度
                self.optimizer.step()  # 更新模型参数

                total_loss += loss.item()
                total_y.extend(batch_y.cpu().numpy())
                total_pred.extend(outputs.detach().cpu().numpy())

                time2 = time.time()
                if count % 100 == 0:
                    # print(outputs)
                    print(f"batch {count} time : {time2-time1}")
                count = count + 1

                # 在进行梯度更新后打印梯度统计
                # print_gradient_statistics(self.model)

            train_metrics = cal_all(total_y, total_pred)
            print(f"Epoch {epoch}: Loss = {total_loss / len(total_y)}, Metrics = {train_metrics}")

            val_metrics = self.eval_model(val_data_path)
            print(f"Validation Metrics: {val_metrics}")

            if val_metrics['mae'] < best_score:
                best_score = val_metrics['mae']
                best_model_path = os.path.join(save_folder, f'model_{epoch}.pth')
                torch.save(self.model.state_dict(), best_model_path)
                print(f"Model improved and saved for epoch {epoch} with MAE: {val_metrics['mae']}")

        # Load the best model for final testing
        if best_model_path is not None and os.path.exists(best_model_path):
            self.model.load_state_dict(torch.load(best_model_path))
            print(f"Loaded best model for final testing from {best_model_path}")

        test_metrics = self.eval_model(test_data_path)
        print(f"Test Metrics: {test_metrics}")

    def eval_model(self, data_path):
        self.model.eval()  # Set model to evaluation mode
        total_val_loss, total_val_y, total_val_pred = 0, [], []
        data_loader = DataLoader()

        for batch_data in data_loader.data_generator(data_path, self.batch_size, self.label_select):
            batch_x, batch_masks, batch_token_type_ids, batch_refs_x, batch_y = batch_data
            batch_x, batch_masks, batch_token_type_ids, batch_y = map(lambda x: x.to(self.device), [batch_x, batch_masks, batch_token_type_ids, batch_y])
            batch_refs_x = batch_refs_x.to(self.device)

            with torch.no_grad():
                outputs = self.model(batch_x, batch_masks, batch_token_type_ids, batch_refs_x)
                loss = self.criterion(outputs, batch_y)

            total_val_loss += loss.item()
            total_val_y.extend(batch_y.cpu().numpy())
            total_val_pred.extend(outputs.detach().cpu().numpy())

        average_loss = total_val_loss / len(total_val_y)
        metric_scores = cal_all(total_val_y, total_val_pred)
        metric_scores['loss'] = average_loss
        return metric_scores

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a model on scientific texts.')
    args = parser.parse_args()