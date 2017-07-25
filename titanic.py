import argparse
import sys

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf
import pandas as pd
import numpy as np
import csv

TITANIC_TRAIN_DATA = r'C:\Users\crisyp\Documents\machine_learning\kaggle\titanic\train.csv'
TITANIC_TEST_DATA = r'C:\Users\crisyp\Documents\machine_learning\kaggle\titanic\test.csv'
TITANIC_SUBMISSION = r'C:\Users\crisyp\Documents\machine_learning\kaggle\titanic\submission.csv'

def convert_to_x_and_y(titanic_df, y_included=True):
    # now remove columns we won't be using in the first instance

    x = titanic_df.drop('Embarked', 1, inplace=False)
    #x = x.drop('Cabin', 1, inplace=False)
    x = x.drop('Ticket', 1, inplace=False)
    x = x.drop('Name', 1, inplace=False)
    x = x.drop('PassengerId', 1, inplace=False)

    # now some transforms, Sexes

    x.loc[titanic_df.Sex == 'male', 'Sex'] = 1
    x.loc[titanic_df.Sex == 'female', 'Sex'] = 0

    #replace nan in age with average age
    x['Age'].fillna(0, inplace=True)
    x['Age'].replace(0.0, x['Age'].mean(), inplace=True)

    #replace cabin to indicate social status (1 or zero)
    x['Cabin'].fillna(0, inplace=True)
    x.loc[x.Cabin != 0, 'Cabin'] = 1
    x['Cabin'] = pd.to_numeric(x['Cabin'])

    # get predictions
    if(y_included):
        y = x['Survived'].values
        y_survived = y.copy()
        y_died = y.copy()
        y_died[y_died == 1] = 2
        y_died[y_died == 0] = 1
        y_died[y_died == 2] = 0

        y = np.column_stack((y_survived, y_died))

        # remove predictions
        x = x.drop('Survived', 1, inplace=False)

        # convert Sex column to numeric
        x['Sex'] = pd.to_numeric(x['Sex'])

        # now convert X to numpy matrix
        x = (x - x.mean()) / (x.max() - x.min())
        x = x.values

        return x, y
    else:
        # convert Sex column to numeric
        x['Sex'] = pd.to_numeric(x['Sex'])

        # now convert X to numpy matrix
        x = (x - x.mean()) / (x.max() - x.min())
        x = x.values
        return x


if __name__ == '__main__':
    # so first load data
    titanic_df = pd.read_csv(TITANIC_TRAIN_DATA)
    titanic_real_test_df  = pd.read_csv(TITANIC_TEST_DATA)
    titanic_df = titanic_df.sample(frac=1)
    titanic_train_df = titanic_df
    titanic_test_df = titanic_df

    x_train, y_train = convert_to_x_and_y(titanic_train_df)
    x_test, y_test = convert_to_x_and_y(titanic_test_df)
    x_real_test = convert_to_x_and_y(titanic_real_test_df, y_included=False)

    # Create the model
    x = tf.placeholder(tf.float32, [None, 7])
    W = tf.Variable(tf.zeros([7, 2]))
    b = tf.Variable(tf.ones([2]))
    y = tf.matmul(x, W) + b

    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, 2])

    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    sess.run(train_step, feed_dict={x: x_train, y_: y_train})

    # Test trained model
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print("accuracy: ", sess.run(accuracy, feed_dict={x: x_test,
                                        y_: y_test}))
    result = sess.run(tf.argmax(y, 1), feed_dict={x: x_real_test})
    result_y = sess.run(y, feed_dict={x: x_real_test})

    with open(TITANIC_SUBMISSION, 'w', newline='') as csvfile:
        submission_writer = csv.writer(csvfile, delimiter=',')
        submission_writer.writerow(['PassengerId', 'Survived'])
        i = 892
        for pred in result:
            val = 0;
            if(pred == 0):
                val = 1
            elif (pred == 1):
                val = 0
            submission_writer.writerow([i, val ])
            i = i + 1
