import numpy as np
import scipy.io as sio
import tensorflow as tf

def load_init_parameter(filepath):
    init_params = sio.loadmat(filepath)

    return init_params


def load_movies(filepath):
    movies = sio.loadmat(filepath)

    return movies

def load_movie_list(filepath):
    def get_movie_name(idx,*arr): return ' '.join(arr)[:-1]
    movie_list = np.array([get_movie_name(*o.split(" ")) for o in open(filepath, encoding='utf-8')])

    return movie_list

def normalize_ratings(Y, R):
    m, n = Y.shape
    ymean = np.zeros((m,1))
    normedY = np.zeros((m,n))
    for i in range(m):
        idx = R[i,:]
        ymean[i] = Y[i,:][idx].mean()
        normedY[i,:][idx] = Y[i,:][idx] - ymean[i]

    return normedY, ymean

def normalize_ratings_v2(Y, R):
    ymean = Y[R != 0].mean()
    normedY = np.zeros(Y.shape)
    normedY[R != 0] = Y[R != 0] - ymean
    return normedY, ymean


movie_list = load_movie_list('./movie_ids.txt')
print(movie_list[0])

ratings = load_movies('./ex8_movies.mat')
Y, R = ratings['Y'], ratings['R']
normed_Y, ymean = normalize_ratings_v2(Y, R)

init_params = load_init_parameter('./ex8_movieParams.mat')
X_init = init_params['X']
Theta_init = init_params['Theta']

X_init_sample = X_init[:5,:3]
Theta_init_sample = Theta_init[:4,:3]
Y_sample = Y[:5,:4]
R_sample = R[:5,:4]

X_sample = tf.Variable(X_init_sample, trainable=True, name='X_sample')
Theta_sample = tf.Variable(Theta_init_sample, trainable=True, name='Theta_sample')

with tf.GradientTape() as tape:
    Theta_sample_t = tf.transpose(Theta_sample)
    logits = tf.matmul(X_sample, Theta_sample_t) * R_sample
    total_ratings_tmp = R_sample.sum()
    loss = 0.5 * tf.math.reduce_sum(tf.square(logits-Y_sample))

print(loss.numpy())




num_users, num_movies, num_features = init_params['num_users'].item(0), init_params['num_movies'].item(0), init_params['num_features'].item(0)

X = tf.Variable(X_init, trainable=True, name='X')
Theta = tf.Variable(Theta_init, trainable=True, name='Theta')
bias_u = tf.Variable(np.zeros((1, num_users)), trainable=True, name='bias_u')
bias_i = tf.Variable(np.zeros((num_movies,1)), trainable=True, name='bias_i')
optimizer = tf.optimizers.RMSprop(0.01)

def regulate_variable(var):
    return tf.math.reduce_sum(tf.math.square(var))

def train_step(X, Theta, normed_Y, R, lamb=1):

    with tf.GradientTape(persistent=True) as tape:
        tape.watch(X)
        tape.watch(Theta)
        Theta_t = tf.transpose(Theta)
        logits = (tf.matmul(X, Theta_t) + bias_u + bias_i) * R
        total_ratings = R.sum()
        loss = tf.math.reduce_sum(tf.math.square(normed_Y - logits)) #* 1/total_ratings
        loss += lamb * (regulate_variable(X) + regulate_variable(Theta) + regulate_variable(bias_u) + regulate_variable(bias_i))
        loss *= (1/total_ratings)

    X_grad = tape.gradient(loss, X)
    Theta_grad = tape.gradient(loss, Theta)
    bias_u_grad = tape.gradient(loss, bias_u)
    bias_i_grad = tape.gradient(loss, bias_i)
    optimizer.apply_gradients(zip([X_grad], [X]))
    optimizer.apply_gradients(zip([Theta_grad], [Theta]))
    optimizer.apply_gradients(zip([bias_u_grad],[bias_u]))
    optimizer.apply_gradients(zip([bias_i_grad], [bias_i]))

    return loss

epochs = 100
loss_prev = 1e8
for epoch in range(epochs):
    
    loss = train_step(X,Theta, normed_Y, R)
    if epoch%50 == 0:
        print("epoch: {}, loss: {}".format(epoch+1, loss))

    #if (loss_prev-loss) < 1e-3:
    #    print("Early Stopping on epoch {}".format(epoch))
    #    break

    loss_prev = loss


prediction = tf.matmul(X, tf.transpose(Theta)).numpy() + bias_u.numpy() + bias_i.numpy()
prediction += ymean
print(ymean)
print(bias_i.numpy()[:5], bias_u.numpy()[:,:5])
user_id = 0
top10_ratings = np.sort(prediction[user_id,:])[::-1][:10]
top10_idx = np.argsort(prediction[user_id, :])[::-1][:10]

print("user_id {}'s top10 recommendation".format(user_id+1))
for i in range(10):
    print("{} {}, predicted rating - {}".format(top10_idx[i],movie_list[top10_idx[i]],top10_ratings[i]))
