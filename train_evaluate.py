from __future__ import division, print_function, absolute_import

import os
import numpy as np
import tensorflow as tf
import tqdm
from datasets import SamplePipeline
import models_factory
from optimizers import get_optimizer
import click

#args
@click.command()
##Data args
@click.option("--data_path", default="./TFRecords")
@click.option("--n_classes", default=100)
##Training args
@click.option('--model_name', default='resnet18')
@click.option("--batch_size", default=50)
@click.option("--epochs", default=20)
@click.option("--lr", default=0.003)
@click.option("--weight_decay", default=0.0)
@click.option("--augment", is_flag=True)
@click.option("--opt", default="SGD")
##logging args
@click.option("-o","--base_log_dir", default="logs")

def main(data_path, model_name, n_classes,batch_size,
        epochs,lr,weight_decay, augment, opt, base_log_dir):

    #Fix TF random seed
    tf.random.set_seed(1777)
    log_dir = os.path.join(os.path.expanduser(base_log_dir),
                           "{}".format(model_name))
    os.makedirs(log_dir, exist_ok=True)

    # dataset
    data_reader = SamplePipeline()
    files = tf.io.gfile.glob("{}/*.tfrecord".format(os.path.expanduser(data_path)))
    print("Found {} tfrecords in {}".format(len(files), data_path))
    #split the data to train and validation
    train_samples = int(len(files)*0.8)
    train_files = files[:train_samples]
    valid_files = files[train_samples:]
    train_dataset = data_reader.input_pipeline(train_files, batch_size, augmentation=augment)
    valid_dataset = data_reader.input_pipeline(valid_files, batch_size, evaluate=True)

    #Network

    model = models_factory.get_model(\
            model_name,num_classes=n_classes,weight_decay=weight_decay)

    #Train optimizer, loss
    #nrof_steps_per_epoch = (train_samples//batch_size)
    #boundries = [nrof_steps_per_epoch*75, nrof_steps_per_epoch*125]
    #values = [lr, lr*0.1, lr*0.01]
    #lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(\
    #                boundries,
    #                values)
    optimizer = get_optimizer(opt, lr)
    #tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.9)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

    #metrics
    train_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
    valid_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()

    #Train step
    @tf.function
    def train_step(x,labels):
        with tf.GradientTape() as t:
            logits = model(x, training=True)
            loss = loss_fn(labels, logits)

        gradients = t.gradient(loss,model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss, logits

    #Run

    

    #Summary writers
    train_summary_writer = tf.summary.create_file_writer(os.path.join(log_dir,
                                                                      'summaries',
                                                                      'train'))
    test_summary_writer = tf.summary.create_file_writer(os.path.join(log_dir,
                                                                     'summaries',
                                                                     'test'))


    ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, net=model)
    ckpt_path = os.path.join(log_dir, 'checkpoints')
    manager = tf.train.CheckpointManager(ckpt,ckpt_path, max_to_keep=3)
    ckpt.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        print("Restored from {}".format(manager.latest_checkpoint))

    else:
        print("Initializing from scratch.")
    ep_cnt = tf.Variable(initial_value=0, trainable=False, dtype=tf.int64)


    for ep in tqdm.trange(epochs, desc='Epoch Loop'):
        if ep < ep_cnt:
            continue

        # update epoch counter
        ep_cnt.assign_add(1)
        with train_summary_writer.as_default():
            # train for an epoch
            for step, (x,y) in enumerate(train_dataset):
                if len(x.shape)==3:
                    x = tf.expand_dims(x,3)
                tf.summary.image("input_image", x, step=optimizer.iterations)
                loss, logits = train_step(x,y)
                train_acc_metric(y, logits)
                ckpt.step.assign_add(1)
                tf.summary.scalar("loss", loss, step=optimizer.iterations)
                
                if int(ckpt.step) % 1000 == 0:
                    save_path = manager.save()
                    print("Saved checkpoint for step {}: {}".format(int(ckpt.step),
                                                                    save_path))
                # Log every 200 batch
                if step % 200 == 0:
                    train_acc = train_acc_metric.result() 
                    print("Training loss {:1.2f}, accuracu {} at step {}".format(\
                            loss.numpy(),
                            float(train_acc),
                            step))


            ## Display metrics at the end of each epoch.
            train_acc = train_acc_metric.result()
            tf.summary.scalar("accuracy", train_acc, step=optimizer.iterations)
            print('Training acc over epoch: %s' % (float(train_acc),))
            # Reset training metrics at the end of each epoch
            train_acc_metric.reset_states()
 

    ############################## Test the model #############################
        with test_summary_writer.as_default():
            for x_batch, y_batch in valid_dataset:
                if len(x_batch.shape)==3:
                    x_batch = tf.expand_dims(x_batch, 3)
                test_logits = model(x_batch, training=False)
                # Update test metrics
                valid_acc_metric(y_batch, test_logits)

            valid_acc = valid_acc_metric.result()
            tf.summary.scalar("accuracy", valid_acc, step=optimizer.iterations)
            valid_acc_metric.reset_states()
            print('[Epoch {}] Valid acc: {}'.format(ep, float(valid_acc)))

    save_path = manager.save()
    print("Saved checkpoint for step {}: {}".format(int(ckpt.step),
                                                save_path))

if __name__=="__main__":
    main()
