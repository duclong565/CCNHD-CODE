import numpy as np
import pandas as pd
from math import log2
import graphviz

# Dữ liệu mẫu
data = {
    'Outlook': ['Sunny', 'Sunny', 'Overcast', 'Rain', 'Rain', 'Rain', 'Overcast', 'Sunny', 'Sunny', 'Rain', 'Sunny',
                'Overcast', 'Overcast', 'Rain'],
    'Temperature': ['Hot', 'Hot', 'Hot', 'Mild', 'Cool', 'Cool', 'Cool', 'Mild', 'Cool', 'Mild', 'Mild', 'Mild', 'Hot',
                    'Mild'],
    'Humidity': ['High', 'High', 'High', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'Normal', 'Normal',
                 'High', 'Normal', 'High'],
    'Wind': ['Weak', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong', 'Strong',
             'Weak', 'Strong'],
    'Play': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No']
}

df = pd.DataFrame(data)


# Hàm tính entropy
def entropy(y):
    classes, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)
    return -sum(p * log2(p) for p in probabilities)


# Hàm tính information gain
def information_gain(X, y, feature):
    total_entropy = entropy(y)
    values, counts = np.unique(X[feature], return_counts=True)
    weighted_entropy = sum(counts[i] / len(y) * entropy(y[X[feature] == values[i]]) for i in range(len(values)))
    return total_entropy - weighted_entropy


# Hàm tính gain ratio (cho C4.5)
def gain_ratio(X, y, feature):
    ig = information_gain(X, y, feature)
    split_info = entropy(X[feature])
    return ig / split_info if split_info != 0 else 0


# Hàm tính Gini index (cho CART)
def gini_index(y):
    classes, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)
    return 1 - sum(p ** 2 for p in probabilities)


# ID3 Algorithm
def id3(X, y, features):
    if len(np.unique(y)) == 1:
        return np.unique(y)[0]
    if len(features) == 0:
        return np.argmax(np.bincount(y))

    best_feature = max(features, key=lambda f: information_gain(X, y, f))
    tree = {best_feature: {}}

    for value in np.unique(X[best_feature]):
        sub_X = X[X[best_feature] == value].drop(best_feature, axis=1)
        sub_y = y[X[best_feature] == value]
        sub_features = [f for f in features if f != best_feature]
        tree[best_feature][value] = id3(sub_X, sub_y, sub_features)

    return tree


# C4.5 Algorithm
def c45(X, y, features):
    if len(np.unique(y)) == 1:
        return np.unique(y)[0]
    if len(features) == 0:
        return np.argmax(np.bincount(y))

    best_feature = max(features, key=lambda f: gain_ratio(X, y, f))
    tree = {best_feature: {}}

    for value in np.unique(X[best_feature]):
        sub_X = X[X[best_feature] == value].drop(best_feature, axis=1)
        sub_y = y[X[best_feature] == value]
        sub_features = [f for f in features if f != best_feature]
        tree[best_feature][value] = c45(sub_X, sub_y, sub_features)

    return tree


# CART Algorithm
def cart(X, y, features):
    if len(np.unique(y)) == 1:
        return np.unique(y)[0]
    if len(features) == 0:
        return np.argmax(np.bincount(y))

    best_feature = min(features, key=lambda f: gini_index(y[X[f] == np.unique(X[f])[0]]) + gini_index(
        y[X[f] == np.unique(X[f])[1]]))
    tree = {best_feature: {}}

    for value in np.unique(X[best_feature]):
        sub_X = X[X[best_feature] == value].drop(best_feature, axis=1)
        sub_y = y[X[best_feature] == value]
        sub_features = [f for f in features if f != best_feature]
        tree[best_feature][value] = cart(sub_X, sub_y, sub_features)

    return tree


# Hàm vẽ cây quyết định
def visualize_tree(tree, name):
    dot = graphviz.Digraph(comment=name)
    dot.attr(rankdir='TB')

    def add_nodes_edges(node, parent=None):
        if isinstance(node, dict):
            for key, value in node.items():
                if parent:
                    dot.edge(parent, key)
                add_nodes_edges(value, key)
        else:
            if parent:
                dot.edge(parent, str(node))
            dot.node(str(node), str(node), shape='box')

    add_nodes_edges(tree)
    dot.render(name, format='png', cleanup=True)
    print(f"{name}.png has been generated.")


# Chạy các thuật toán và vẽ cây quyết định
features = ['Outlook', 'Temperature', 'Humidity', 'Wind']
X = df[features]
y = df['Play']

id3_tree = id3(X, y, features)
c45_tree = c45(X, y, features)
cart_tree = cart(X, y, features)

visualize_tree(id3_tree, 'id3_tree')
visualize_tree(c45_tree, 'c45_tree')
visualize_tree(cart_tree, 'cart_tree')

print("Trees have been visualized. Check the generated PNG files.")

# Giải thích chi tiết về các thuật toán

"""
Giải thích chi tiết về các thuật toán ID3, C4.5 và CART

1. Thuật toán ID3 (Iterative Dichotomiser 3):
   
   Bước 1: Tính entropy của biến mục tiêu (trong trường hợp này là 'Play').
   Bước 2: Tính information gain cho mỗi đặc trưng.
   Bước 3: Chọn đặc trưng có information gain cao nhất làm nút gốc.
   Bước 4: Tạo các nút con cho mỗi giá trị duy nhất của đặc trưng đã chọn.
   Bước 5: Lặp lại bước 1-4 cho mỗi nút con cho đến khi đạt một trong các điều kiện dừng:
           - Tất cả các mẫu trong một nút thuộc cùng một lớp
           - Không còn đặc trưng nào để chia
           - Nút không có mẫu nào

   Khái niệm chính: Information Gain = Entropy(cha) - Tổng có trọng số của Entropy(con)

2. Thuật toán C4.5 (cải tiến từ ID3):

   Bước 1-4: Giống như ID3
   Bước 5: Tính split information cho mỗi đặc trưng.
   Bước 6: Tính gain ratio cho mỗi đặc trưng (Information Gain / Split Information).
   Bước 7: Chọn đặc trưng có gain ratio cao nhất làm nút gốc.
   Bước 8: Lặp lại bước 1-7 cho mỗi nút con cho đến khi đạt điều kiện dừng.

   Khái niệm chính: Gain Ratio = Information Gain / Split Information
   Điều này giúp giải quyết vấn đề ID3 thiên vị với các đặc trưng có nhiều giá trị duy nhất.

3. Thuật toán CART (Classification and Regression Trees):

   Bước 1: Tính chỉ số Gini cho biến mục tiêu.
   Bước 2: Với mỗi đặc trưng, tính chỉ số Gini cho mỗi điểm chia có thể.
   Bước 3: Chọn đặc trưng và điểm chia cho chỉ số Gini thấp nhất.
   Bước 4: Tạo các nút con dựa trên điểm chia đã chọn.
   Bước 5: Lặp lại bước 1-4 cho mỗi nút con cho đến khi đạt điều kiện dừng.

   Khái niệm chính: Chỉ số Gini = 1 - Tổng(p_i^2), trong đó p_i là xác suất của lớp i.
   Chỉ số Gini thấp hơn chỉ ra độ tinh khiết tốt hơn của các nút.

Sự khác biệt chính:
1. Tiêu chí chia:
   - ID3 sử dụng Information Gain
   - C4.5 sử dụng Gain Ratio
   - CART sử dụng chỉ số Gini

2. Loại đặc trưng:
   - ID3 và C4.5 hoạt động tốt với đặc trưng phân loại
   - CART có thể xử lý cả đặc trưng phân loại và số

3. Cấu trúc cây:
   - ID3 và C4.5 có thể tạo ra các phân chia đa chiều
   - CART chỉ tạo ra các phân chia nhị phân

4. Xử lý giá trị thiếu:
   - C4.5 có phương pháp tích hợp để xử lý giá trị thiếu
   - ID3 và CART thường yêu cầu tiền xử lý để xử lý giá trị thiếu

5. Phòng chống overfitting:
   - C4.5 bao gồm bước cắt tỉa để giảm overfitting
   - CART sử dụng cắt tỉa dựa trên độ phức tạp chi phí
   - ID3 cơ bản không bao gồm cắt tỉa (mặc dù có các phiên bản mở rộng)

Các thuật toán này tạo nền tảng cho việc học cây quyết định trong máy học
và rất quan trọng trong việc hiểu các phương pháp ensemble nâng cao hơn như Random Forests
và Gradient Boosting Machines.
"""