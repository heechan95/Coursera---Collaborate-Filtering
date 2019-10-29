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
    
    ymean = Y[R != 0].mean()
    ynorm = Y[R != 0].mean()

    normedY = (Y - ymean) / ynorm

    return normedY, ymean, ynorm


movie_list = load_movie_list('./movie_ids.txt')
print(movie_list[0])

ratings = load_movies('./ex8_movies.mat')
Y, R = ratings['Y'], ratings['R']
normed_Y, ymean, ynorm = normalize_ratings(Y, R)

init_params = load_init_parameter('./ex8_movieParams.mat')
X_init = init_params['X']
Theta_init = init_params['Theta']

num_users, num_movies, num_features = init_params['num_users'].item(0), init_params['num_movies'].item(0), init_params['num_features'].item(0)

X = tf.Variable(X_init, trainable=True, name='X')
Theta = tf.Variable(Theta_init, trainable=True, name='Theta')

optimizer = tf.optimizers.RMSprop(0.01)

def train_step(X, Theta, normed_Y, R):

    with tf.GradientTape(persistent=True) as tape:
        tape.watch(X)
        tape.watch(Theta)
        Theta_t = tf.transpose(Theta)
        logits = tf.matmul(X, Theta_t) * R
        total_ratings = R.sum()
        loss = tf.math.reduce_sum(tf.math.square(normed_Y - logits)) * 1/total_ratings

    X_grad = tape.gradient(loss, X)
    Theta_grad = tape.gradient(loss, Theta)
    optimizer.apply_gradients(zip([X_grad], [X]))
    optimizer.apply_gradients(zip([Theta_grad], [Theta]))

    return loss

epochs = 100
loss_prev = 1e7
for epoch in range(epochs):
    
    loss = train_step(X,Theta, normed_Y, R)
    if epoch%10 == 0:
        print("epoch: {}, loss: {}".format(epoch+1, loss))

    if (loss_prev-loss) < 1e-3:
        print("Early Stopping on epoch {}".format(epoch))
        break

    loss_prev = loss


prediction = tf.matmul(X, tf.transpose(Theta)).numpy()
prediction += ymean

user_id = 0
top10_ratings = np.sort(prediction[user_id,:])[::-1][:10]
top10_idx = np.argsort(prediction[user_id, :])[::-1][:10]

print("user_id {}'s top10 recommendation".format(user_id+1))
for i in range(10):
    print("{} {}, predicted rating - {}".format(top10_idx[i],movie_list[top10_idx[i]],top10_ratings[i]))
