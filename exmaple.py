import pandas as pd
import tensorflow as tf
import numpy as np
data = pd.read_csv('abalone.txt')
#类似C#的Idataview，默认将第一行识别为特征名称

data['Sex'] = data['Sex'].apply(lambda x: 0 if x == 'F' else (1 if x == 'M' else 2))

Input = data[['Length', 'Diameter', 'Height', 'Whole', 'Shucked', 'Viscera', 'Shell', 'Rings']
]

Output = data['Sex']

model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(8,)),
#64个神经元，激活函数为relu，输入向量长度为8
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')  # 使用softmax激活函数进行多类别分类
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
#
model.fit(Input, Output, epochs=1000)
#接受2个numpy数组，训练100次
# 评估模型
(test_loss, test_accuracy) = model.evaluate(Input, Output)
print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")

new_data = pd.DataFrame({'Length': [0.53],
                         'Diameter': [0.415],
                         'Height': [0.15],
                         'Whole': [0.7775],
                         'Shucked': [0.237],
                         'Viscera': [0.1415],
                         'Shell': [0.33],
                         'Rings': [20]})
# 进行预测
predictions = model.predict(new_data)  # 使用模型进行预测
print((predictions))
#与输出标签长度相同的数组，是对应标签的概率分布
#三个可能的性别类别（'F'、'M'和'I'）。每个值表示模型对样本属于相应类别的置信度（概率）

class_labels = ['F', 'M', 'I']
predicted_class_index = np.argmax(predictions, axis=-1)
predicted_gender = [class_labels[i] for i in predicted_class_index]
print(f"Predicted Gender: {predicted_gender}")
