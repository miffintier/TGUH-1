import numpy as np
text_array=np.load("/home/xx/Projects/TGUH/clip_feature/flickr_train_img.npy")

A=text_array
A_T = A.T
 
# 计算A与A_T的点积，得到未归一化的gram矩阵
dot_product = np.dot(A, A_T)
 
# 计算A中每个向量的范数（即向量的模长）
norms = np.linalg.norm(A, axis=1, keepdims=True)
norms_expanded = norms * norms.T
cosine_similarity_matrix = dot_product / norms_expanded
 
# 打印余弦相似度矩阵
print("Cosine Similarity Matrix:")
print(cosine_similarity_matrix)
print(cosine_similarity_matrix.shape)

A=np.save("flickr_img_S",cosine_similarity_matrix)