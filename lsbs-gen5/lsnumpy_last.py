import numpy as np
import matplotlib.pyplot as plt
# 난수 생성하여 3x3 크기의 행렬 생성

np.random.seed(2024)
img1 = np.random.rand(3, 3)

img1

# 행렬을 이미지로 표시
plt.figure(figsize=(10, 5))  # (가로, 세로) 크기 설정
plt.imshow(img1, cmap='gray', interpolation='nearest');
plt.colorbar();
plt.show();

img_mat = np.loadtxt('C:\\Users\\kbk29\\OneDrive\\바탕 화면\\Workspace\\LS_BigDataSchool\\Data\\img_mat.csv', delimiter=',', skiprows=1)

# 이미지 값의 최대 최소 값 찾기
img_mat.max(), img_mat.min()

# 사진을 하기 위해서 0~1사이의 값으로 만듦
# 행렬 값을 0과 1 사이로 변환
img_mat = img_mat / 255.0
import matplotlib.pyplot as plt
# 행렬을 이미지로 변환하여 출력
plt.imshow(img_mat, cmap='gray', interpolation='nearest');
plt.colorbar();
plt.show();

# 5행 2열의 행렬 생성
x = np.arange(1, 11).reshape((5, 2)) * 2
print("원래 행렬 x:\n", x)

transposed_x = x.transpose()
print("전치된 행렬 x:\n", transposed_x)

x.shape, transposed_x.shape

# 밝기 조절 하는 방법

# 전체에 0.2를 더한다 
# 2단계 1이상의 값을 가지는 애들 -> 1로변환
img = img_mat / 255.0
img = img + 0.2
#필터링
img[img > 1.0] = 1.0
plt.imshow(img, cmap='gray', interpolation='nearest');
plt.colorbar();
plt.show();

# 2행 3열의 행렬 y 생성 np.dot은 내적이고 외적은 cross이다.
y = np.arange(1, 7).reshape((2, 3))
print("행렬 y:\n", y)

print("행렬 x, y의 크기:", x.shape, y.shape)

dot_product = x.dot(y)
print("행렬곱 x * y:\n", dot_product)

import numpy as np

mat_A = np.array([1, 2, 4, 3], dtype=int).reshape(2, 2)
mat_B = np.array([2, 1, 3, 1], dtype=int).reshape(2, 2)

result = mat_A.dot(mat_B)  # 또는 mat_A @ mat_B 또는 np.matmul(mat_A, mat_B)
print(result)

res = mat_A @ mat_B
print(res)

import imageio

hi = imageio.imread("C:\\Users\\kbk29\\OneDrive\\바탕 화면\\Workspace\\LS_BigDataSchool\\hi.png")

print('이미지 클래스:',type(hi))
print('이미지 차원', hi.shape)

import matplotlib.pyplot as plt

plt.imshow(hi);
plt.axis('off')
plt.show()

# 흑백으로 변환
hi = np.mean(hi[:, :, :3], axis = 2)
plt.imshow(hi, cmap='gray');
plt.axis('off');
plt.show();

#넘파이 행렬 연습 문제 풀기

A = np.arange(1,5).reshape(2,2)
B = np.arange(5,9).reshape(2,2)

A @ B # A.dot(B) or np.matmul(A, B)

A_arr = np.arange(1,7 ).reshape(2,3)
B_arr = np.arange(7,13).reshape(3,2)

np.matmul(A_arr, B_arr)

A = np.arange(2,6).reshape(2,2)
I = np.array([[1,0],[0,1]], dtype= int)
A @ I

A = np.arange(1,5).reshape(2,2)
Z = np.zeros((2, 2))
np.matmul(A,Z)

D = np.array([[2,0],[0,3]])
A = np.arange(4,8).reshape(2,2)
D@A

A = np.arange(1,7).reshape(3,2)
V= np.array([[0.4],[0.6]])
A@V

A = np.arange(1,5).reshape(2,2)
B = np.arange(5,9).reshape(2,2)
C = np.arange(9,13).reshape(2,2)

T = np.array([A, B])
T @ C

#문제의 의도 2배씩 대각원소들이 증가하는 구조를 의미하는것을 의미하는데

#문제의 의도는 멱틍행렬을 바란것이었으나 되지 않았음
S = np.array([[2,-1],[-1,2]])
S*S
S*S*S
inv_S = np.linalg.inv(S)
inv_S
S@inv_S
inv_S @ inv_S 

A = np.arange(1,5).reshape(2,2)
B = np.arange(5,9).reshape(2,2)
C = np.arange(9,13).reshape(2,2)

(A@B)@C
A@(B@C)
B@C@A
C@A@B

A = np.array([[3,2,-1], [ 2,-2,4],[-1,0.5,-1]])
b = np.array([[1],[-2],[0]])
inv_A = np.linalg.inv(A)
x = inv_A@b
x

A= np.array([[1,2,5],[3,4,2],[5,6,1]])
V = np.array([[0.3],[0.3],[0.4]])

np.matmul(A,V)


import pandas as pd
# 데이터 프레임 생성
df = pd.DataFrame({
    'col1': ['one', 'two', 'three', 'four', 'five'],
    'col2': [6, 7, 8, 9, 10]
})
print(df)