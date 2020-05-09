
# model instantiation -----------------------------------------------
# set defaul flags
FLAGS <- flags(
  flag_numeric("dropout", 0.4),
  flag_numeric("lambda", 0.01),
  flag_numeric("lr", 0.01),
  flag_numeric("bs", 100)
)

# model configuration
model <- keras_model_sequential() %>%
  layer_dense(units = 128, input_shape = V, activation = "relu", name = "layer_1",
              kernel_regularizer = regularizer_l2(FLAGS$lambda)) %>%
  layer_dropout(rate = FLAGS$dropout) %>%
  layer_dense(units = 64, activation = "relu", name = "layer_2",
              kernel_regularizer = regularizer_l2(FLAGS$lambda)) %>%
  layer_dropout(rate = FLAGS$dropout) %>%
  layer_dense(units = 32, activation = "relu", name = "layer_3",
              kernel_regularizer = regularizer_l2(FLAGS$lambda)) %>%
  layer_dropout(rate = FLAGS$dropout) %>%
  layer_dense(units = ncol(yc_train), activation = "softmax", name = "layer_out") %>%
  compile(loss = "categorical_crossentropy", metrics = "accuracy",
          optimizer = optimizer_adam(lr = FLAGS$lr),
  )

fit <- model %>% fit(
  x = x1_train, y = yc_train,
  validation_data = list(x1_val, yc_val),
  epochs = 100,
  batch_size = FLAGS$bs,
  verbose = 0,
  callbacks = callback_early_stopping(monitor = "val_accuracy", patience = 20)
)

# store accuracy on test set for each run
score <- model %>% evaluate(
  x1_test, yc_test,
  verbose = 0
)