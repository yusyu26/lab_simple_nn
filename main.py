import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

import numpy as np
import matplotlib.pyplot as plt
import random

import yaml
import os

def set_seed(seed):
    """
    乱数シードを固定するための関数
    スクリプトを複数回実行した際に同じ結果が出るようにするための設定を行う。
    """
    torch.manual_seed(seed)          # PyTorchのCPUにおける乱数シード
    torch.cuda.manual_seed(seed)     # PyTorchのGPUにおける乱数シード（GPUを使う場合）
    torch.backends.cudnn.deterministic = True # cuDNNの決定論的モードを有効化（GPUを使う場合）
    torch.backends.cudnn.benchmark = False    # cuDNNの自動チューニングを無効化（GPUを使う場合）
    np.random.seed(seed)             # NumPyの乱数シード
    random.seed(seed)                # Python標準の乱数シード

def load_mnist_data(batch_size=64, target_labels=[0, 1]):
    transform = transforms.Compose([
        transforms.ToTensor(),
        #   - MNISTデータセット全体の平均値と標準偏差を使用
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # train=True: 訓練データセットをロードします。
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    # train=False: テストデータセットをロードします。
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)

    # 指定したラベルのデータのみを抽出するためのインデックスリストを作成
    # enumerate(train_dataset.targets): データセットの各要素のインデックスとラベルを取得
    # if label in target_label: ラベルが指定したラベルの場合のみ抽出
    train_indices_filtered = [i for i, label in enumerate(train_dataset.targets) if label in target_labels]
    test_indices_filtered = [i for i, label in enumerate(test_dataset.targets) if label in target_labels]

    train_subset = Subset(train_dataset, train_indices_filtered)
    test_subset = Subset(test_dataset, test_indices_filtered)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)

    num_classes = len(target_labels)  # 抽出されたクラスの数を計算

    print(f"[INFO] Training data ({target_labels[0]} and {target_labels[1]}): {len(train_loader.dataset)} samples")
    print(f"[INFO] Test data ({target_labels[0]} and {target_labels[1]}): {len(test_loader.dataset)} samples")

    return train_loader, test_loader, num_classes

# ニューラルネットワークモデルの定義
class TestNN(nn.Module):
    """
    MNISTの2クラス分類（0と1）を行うためのシンプルな全結合ニューラルネットワーク。
    """
    def __init__(self, hidden_nodes=128, num_classes=2):
        super(TestNN, self).__init__()
        self.fc1 = nn.Linear(28 * 28, hidden_nodes)  # 入力層 784ノード
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_nodes, num_classes) # 出力層 2ノード（0と1の分類）

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # 画像をフラット化
        x = self.fc1(x)          # 入力層から隠れ層へ
        x = self.relu(x)         # ReLU活性化関数
        x = self.fc2(x)          # 隠れ層から出力層へ

        return x

def train(model, device, train_loader, optimizer, criterion, epoch, label_mapping):
    """
    モデルを訓練するための関数。
    各エポックで、訓練データセット全体を処理します。
    """
    model.train() # モデルを訓練モードに設定

    running_loss = 0.0 # 訓練中の累積損失を追跡
    correct = 0        # 正しく分類されたサンプルの数を追跡
    total = 0          # 処理された総サンプル数を追跡

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        target_mapped = torch.zeros_like(target) # targetと同じ形状の0で埋められたテンソルを作成
        for original_label, mapped_index in label_mapping.items():
            target_mapped[target == original_label] = mapped_index
        target = target_mapped # マッピングされたターゲットを使用

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        # 予測されたクラスの取得（最も高いスコアを持つインデックス）
        # torch.max(output.data, 1) は、outputテンソルの次元1（クラス次元）に沿って最大値とインデックスを返す
        # _ は最大値、predicted はそのインデックス（クラスID）
        _, predicted = torch.max(output.data, 1)
        total += target.size(0) # バッチサイズを総サンプル数に加算
        correct += (predicted == target).sum().item() # 予測と正解が一致した数を加算

        # 定期的に訓練の進捗を出力
        if batch_idx % 100 == 0: # 100バッチごとにログを出力
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
    # エポック終了時の訓練サマリー
    avg_loss = running_loss / len(train_loader) # エポック全体の平均損失
    accuracy = 100. * correct / total           # エポック全体の精度
    print(f'\nEpoch {epoch} Training Summary: Avg Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')


def test(model, device, test_loader, criterion, label_mapping):
    """
    モデルを評価するための関数。
    テストデータセット全体でのモデルの性能を測定します。
    """
    model.eval() # モデルを評価モードに設定

    test_loss = 0.0 # テスト中の累積損失を追跡
    correct = 0     # 正しく分類されたサンプルの数を追跡

    # 勾配計算を無効化
    # 評価時にはパラメータの更新は行わないので、勾配計算は不要。
    # これによりメモリ使用量を節約し、計算を高速化します。
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            target_mapped = torch.zeros_like(target)
            for original_label, mapped_index in label_mapping.items():
                target_mapped[target == original_label] = mapped_index
            target = target_mapped # マッピングされたターゲットを使用

            output = model(data)
            test_loss += criterion(output, target).item() # 損失を加算 (item()でテンソルから数値を取り出す)

            _, predicted = torch.max(output.data, 1)
            correct += (predicted == target).sum().item()

    # データセットの枚数で割って平均損失を計算
    # test_loss はバッチごとの損失の合計なので、データセット全体のサンプル数で割ることで平均損失を出す
    test_loss /= len(test_loader.dataset) # len(test_loader.dataset) はテストデータの総サンプル数

    # 精度を計算
    accuracy = 100. * correct / len(test_loader.dataset)

    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n')
    return accuracy, test_loss

def main():
    # 設定ファイルの読み込み
    config_path = 'config.yaml'
    if not os.path.exists(config_path):
        print(f"Error: Config file '{config_path}' not found. Please ensure it's mounted or copied.")
        return

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    print(f"[INFO] Loaded configuration from '{config_path}':\n{config}")

    # 設定値を変数に割り当てる
    global_seed = config['global_seed']
    batch_size = config['data']['batch_size']
    target_labels = config['data']['target_labels']
    hidden_nodes = config['model']['hidden_nodes']
    epochs = config['training']['epochs']
    learning_rate = config['training']['learning_rate']

    ## >>> [変更] ラベルマッピング辞書を作成
    # 例: target_labels=[1, 7] なら label_mapping = {1: 0, 7: 1} となる
    label_mapping = {original_label: new_index for new_index, original_label in enumerate(target_labels)}

    set_seed(global_seed)  # 乱数シードを固定

    # GPUが利用可能であれば 'cuda' (NVIDIA GPU), なければ 'cpu' を使用
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # MNISTデータセットのロード
    train_loader, test_loader, num_classes = load_mnist_data(batch_size=batch_size, target_labels=target_labels)

    # モデルの初期化
    model = TestNN(hidden_nodes=hidden_nodes, num_classes=num_classes).to(device)

    # 損失関数と最適化器の定義
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # エポック
    print(f"\n--- Starting training for {epochs} epochs ---")
    for epoch in range(1, epochs + 1): # エポックは1から始まるようにする
        train(model, device, train_loader, optimizer, criterion, epoch, label_mapping)
        # 各エポックの終わりにテストデータで評価
        test(model, device, test_loader, criterion, label_mapping)

    print("\n--- Training finished! ---")

    model.eval()

    random.seed(None)
    num_test_samples = len(test_loader.dataset)
    random_sample_idx = random.randint(0, num_test_samples - 1)  # ランダムなサンプルのインデックスを選択

    sample_image_tensor, sample_true_label = test_loader.dataset[random_sample_idx]

    # モデルに予測させるためにテンソルの形状を調整し、デバイスに送る
    # unsqueeze(0): バッチ次元を追加 (モデルは通常バッチ入力[{1, N}]を期待するため)
    # 例: (1, 28, 28) -> (1, 1, 28, 28) または (28, 28) -> (1, 28, 28)
    # MNISTは(1, 28, 28)なので、(1, 1, 28, 28)にするために unsqueeze(0) を使う
    # to(device): データをモデルと同じデバイスに送る
    sample_image_for_model = sample_image_tensor.unsqueeze(0).to(device)

    # 推論の実行（勾配計算は不要なので no_grad() ブロック内で行う）
    with torch.no_grad():
        # モデルが予測スコア（ロジット）を出力
        output = model(sample_image_for_model)

        # Softmax関数を適用して各クラスの確率に変換
        # dim=1: クラスの次元（出力が2ノードなので、その次元）に対してsoftmaxを適用
        probabilities = torch.softmax(output, dim=1)

        # 最も確率が高いクラスのインデックス（予測ラベル）を取得
        # argmax(dim=1): 指定された次元で最大値のインデックスを返す
        predicted_label_mapped = torch.argmax(probabilities, dim=1).item()
        # .item(): 1要素のテンソルからPythonの数値を取り出す

    # 可視化のために、真のラベルと予測ラベルを元の数字に戻す
    inverse_label_map = {v: k for k, v in label_mapping.items()}
    display_true_label = sample_true_label
    display_predicted_label = inverse_label_map[predicted_label_mapped]

    print(f"\n[SAMPLE] Randomly picked test sample (Index: {random_sample_idx})")
    print(f"         → True Label: {display_true_label}, Predicted Label: {display_predicted_label}")

    # Matplotlib で可視化
    plt.figure(figsize=(3,3)) # 画像のサイズを設定

    display_image = sample_image_tensor.cpu().squeeze().numpy()

    # transforms.Normalize で適用した標準化を元に戻す
    # ピクセル値 = (標準化されたピクセル値 * 標準偏差) + 平均
    # MNISTの標準偏差0.3081、平均0.1307を使って元に戻す
    display_image = display_image * 0.3081 + 0.1307

    # imshow: 画像を表示 (cmap='gray' でグレースケールとして表示)
    plt.imshow(display_image, cmap='gray')

    # タイトルに真のラベルと予測ラベルを表示
    plt.title(f"True: {sample_true_label} / Pred: {display_predicted_label}", fontsize=14)
    plt.axis('off') # 軸（目盛り）を非表示にする

    # 画像をファイルとして保存
    save_path = "output/nn_prediction.png"
    plt.savefig(save_path, bbox_inches='tight') # bbox_inches='tight' で余白をなくす
    print(f"[INFO] Predicted sample image saved as '{save_path}'")

    # 画像を表示 (Dockerコンテナでは通常GUIは表示されないが、エラーは出ない)
    # DockerのGUI環境設定やXサーバー転送がなければ、この行は特に効果はありません。
    # しかし、ファイルを保存しているので問題ありません。
    plt.show()


if __name__ == "__main__":
    main()