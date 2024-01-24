// ############################################################
// ############################################################
// DEFINING MAIN FUNCTIONS
// ############################################################
// ############################################################
let webcam; 
let mobilenet;
let model;
let obj1, obj2, obj3, train_set_num_samples;

const original_images = []; // SAVE IMAGES FOR DISPLAY

let experiment_count = 0; 

const dataset = new WEBCAMDataset(); // TRAINING DATASET
const val_dataset = new WEBCAMDataset(); // VALIDATION DATASET
let val_dataset_avaliable = false, use_update_loop = false; // VALIDATION AVALIABLE OR NOT

var leftSamples=0, rightSamples=0, leftSamplesVal=0, rightSamplesVal=0; // FOR DATA COLLECTION COUNTER
var realleftSamples=0, realrightSamples=0, realleftSamplesVal=0, realrightSamplesVal=0; // FOR DATA COLLECTION COUNTER

let isPredicting = false; // LIVE PREDICTION TAG - STARTS WITH FALSE

var isArray = Array.isArray || function(value) { // FOR SHUFFLE FUNCTION
  return {}.toString.call(value) !== "[object Array]"
};

// ############################################################
// ############################################################
// MODELING AND TRAINING FUNCTIONS
// ############################################################
// ############################################################

async function loadMobilenet() { // LOADS THE MOBILENET MODEL FOR TRANSFER LEARNING
  const mobilenet = await tf.loadLayersModel('https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json');
  const layer = mobilenet.getLayer('conv_pw_13_relu');
  return tf.model({inputs: mobilenet.inputs, outputs: layer.output});
};

function set_split_condition(val_dataset, experiment_count) {
  if (experiment_count == 0) {
    if (val_dataset.xs == null) {
      use_update_loop = true;
    } else {
      use_update_loop = false;
    }
  }
};

async function train() { // TRAINS A MODEL USING WEBCAM COLLECTED IMAGES AND CREATES VISOR DATA
  if (dataset.xs == null) {
    throw new Error('Add some examples before training!');
  }

  showExamples(original_images);  

  const run_name = document.getElementById("training_name").value;

  dataset.ys = null;
  val_dataset.ys = null;
  dataset.encodeLabels(2);
  val_dataset.encodeLabels(2);

  set_split_condition(val_dataset, experiment_count);

  console.log(realleftSamples);
  if (use_update_loop === true) {
    obj1 = dataset.xs.arraySync();
    obj2 = dataset.ys.arraySync();
    obj3 = dataset["labels"];
    shuffle(obj1, obj2, obj3);


    train_set_num_samples = Math.round(0.8*obj3.length);
    dataset.xs = tf.tensor4d(obj1.slice(0, train_set_num_samples));  
    dataset.ys = tf.tensor2d(obj2.slice(0, train_set_num_samples));
    dataset.labels = obj3.slice(0, train_set_num_samples);

    val_dataset.xs = tf.tensor4d(obj1.slice(train_set_num_samples));
    val_dataset.ys = tf.tensor2d(obj2.slice(train_set_num_samples));
    val_dataset.labels = obj3.slice(train_set_num_samples);


    realleftSamples = getOccurrence(obj3.slice(0, train_set_num_samples), 0);
    realrightSamples = getOccurrence(obj3.slice(0, train_set_num_samples), 1); 
    realleftSamplesVal = getOccurrence(obj3.slice(train_set_num_samples), 0); 
    realrightSamplesVal = getOccurrence(obj3.slice(train_set_num_samples), 1); 
    // [realleftSamples, realrightSamples, realleftSamplesVal, realrightSamplesVal] = split_train_test(dataset, val_dataset)
    console.log(realleftSamples);
  }
  console.log(realleftSamples);

  model = tf.sequential({
    layers: [
      tf.layers.flatten({inputShape: mobilenet.outputs[0].shape.slice(1)}),
      tf.layers.dense({ units: parseInt(denseUnitsElement.value), activation: 'relu'}),
      tf.layers.dense({ units: 2, activation: 'softmax'})
    ]
  });

  const optimizer = tf.train.adam(learningRateElement.value);
  model.compile({optimizer: optimizer, loss: 'categoricalCrossentropy', metrics: ['accuracy']});
  let loss = 0;

  // TRAINING MODEL AND LOGGING PARAMETERS
  const surface = { name: "Training Info", tab:run_name};
  const trainLogs = [];
  let trainBatchCount = 0;
  const batchSizeCalc = Math.floor(dataset.xs.shape[0] * batchSizeFractionElement.value);
  if (!(batchSizeCalc > 0)) {
    throw new Error(`Batch size is 0 or NaN. Please choose a non-zero fraction.`);
  }

  const history = await model.fit(dataset.xs, dataset.ys, {
    batchSize: batchSizeCalc,
    epochs: epochsElement.value,
    validationData: [val_dataset.xs, val_dataset.ys],
    callbacks: {
      onBatchEnd: async (batch, logs) => {
        trainBatchCount++;
        // lossStatus('Loss: ' + logs.loss.toFixed(5));
        await tf.nextFrame();
      },
      onEpochEnd: async (epoch, logs) => {
        trainLogs.push(logs);
        // accStatus('Accuracy: ' + logs.acc.toFixed(5))
        tfvis.show.history(surface, trainLogs, ['loss', 'val_loss']);
        tfvis.show.history({name: "Accuracy", tab:run_name}, trainLogs, ['acc', 'val_acc']);
      }
    }
  });

  const labels_for_eval_tensor_train = tf.tensor1d(Object.values(dataset["labels"]));
  const predictions_train = model.predict(dataset.xs);
  const result_array_train = await predictions_train.array();
  const max_values_train = [];
  result_array_train.forEach(function(item){max_values_train.push(indexOfMax(item))});
  const max_values_tensor_train = tf.tensor1d(max_values_train);

  // GETTING LABELS AND PREDICTIONS FOR VALIDATION DATASET
  const labels_for_eval_tensor = tf.tensor1d(Object.values(val_dataset["labels"]));
  const predictions = model.predict(val_dataset.xs);
  const result_array = await predictions.array();
  const max_values = [];
  result_array.forEach(function(item){max_values.push(indexOfMax(item))});
  const max_values_tensor = tf.tensor1d(max_values);

  // GENERATING CONFUSION MATRIX
  const confusion_matrix = await tfvis.metrics.confusionMatrix(labels_for_eval_tensor, max_values_tensor);
  tfvis.render.confusionMatrix({name: 'Confusion Matrix for Validation Set', tab:run_name}, {values: confusion_matrix, tickLabels: ["Left", "Right"]});

  // GENERATING ACCURACY PER CLASS BAR CHART
  const acc_per_class = await tfvis.metrics.perClassAccuracy(labels_for_eval_tensor, max_values_tensor);
  const data_for_bar = [
    { index: "Left", value: acc_per_class[0]["accuracy"] },
    { index: "Right", value: acc_per_class[1]["accuracy"] },
  ];
  tfvis.render.barchart({ name: 'Validation Accuracy per Class', tab:run_name }, data_for_bar);

  // GETING ACCURACY
  const gen_accuracy_train = await tfvis.metrics.accuracy(labels_for_eval_tensor_train, max_values_tensor_train);
  const gen_accuracy = await tfvis.metrics.accuracy(labels_for_eval_tensor, max_values_tensor);
  console.log(realleftSamples);
  const model_metrics_to_append = [run_name, gen_accuracy_train, gen_accuracy, realleftSamples, realrightSamples, realleftSamplesVal, realrightSamplesVal];
  create_result_table(model_metrics_to_append);
  experiment_count++;
};

// ############################################################
// ############################################################
// UTILITY FUNCTIONS
// ############################################################
// ############################################################
function getOccurrence(array, value) { // RETURNS THE NUMBER OF TIMES A GIVEN VALUE IS AT A GIVEN ARRAY
  return array.filter((v) => (v === value)).length;
};

function shuffle() { // SHUFFLES WHATEVER NUMBER OF ARRAYS PASSED INPLACE
  var arrLength = 0;
  var argsLength = arguments.length;
  var rnd, tmp;

  for (var index = 0; index < argsLength; index += 1) {
    if (!isArray(arguments[index])) {
      throw new TypeError("Argument is not an array.");
    }
    if (index === 0) {
      arrLength = arguments[0].length;
    }
    if (arrLength !== arguments[index].length) {
      throw new RangeError("Array lengths do not match.");
    }
  }

  while (arrLength) {
    rnd = Math.floor(Math.random() * arrLength);
    arrLength -= 1;
    for (argsIndex = 0; argsIndex < argsLength; argsIndex += 1) {
      tmp = arguments[argsIndex][arrLength];
      arguments[argsIndex][arrLength] = arguments[argsIndex][rnd];
      arguments[argsIndex][rnd] = tmp;
    }
  }
};

function indexOfMax(arr) { // GETS THE INDEX OF THE MAX ELEMENT IN AN ARRAY
  if (arr.length === 0) {
      return -1;
  }
  var max = arr[0];
  var maxIndex = 0;

  for (var i = 1; i < arr.length; i++) {
      if (arr[i] > max) {
          maxIndex = i;
          max = arr[i];
      }
  }
  return maxIndex;
};

// ############################################################
// ############################################################
// GATHERING RESULTS FUNCTIONS INCLUDING VISUALIZATIONS
// ############################################################
// ############################################################

const headers = ['Run Name', 'Accuracy', 'Val Accuracy', 'Left Samples', 'Right Samples', 'Left Val Samples', 'Right Val Samples'];
const values = [];

function create_result_table(array_to_append) {
  values.push(array_to_append);
  tfvis.render.table({ name: 'Training Iterations', tab: 'All Results' }, {headers, values});
};

function toggle_visor() { // OPENS AND CLOSES TF VISOR
	tfvis.visor().toggle();
};

async function showExamples(images_array) {
  // Create a container in the visor
  const surface = tfvis.visor().surface({ name: 'Input Data Examples', tab: 'Input Data', styles: {height:"90%"}}); 

  const numExamples = images_array.length;
  
  // Create a canvas element to render each example
  for (let i = 0; i < numExamples; i++) {
    // const imageTensor = tf.tidy(() => { // TODO: HOW TO MAKE CROP WORK TO SHOW SMALLER IMAGES
    //   // Reshape the image to 28x28 px
    //   return tf.image.cropAndResize(images_array[i], );
    // });
    const imageTensor = images_array[i];

    const canvas = document.createElement('canvas');
    canvas.width = 20;
    canvas.height = 20;
    canvas.style = 'margin: 4px;';
    await tf.browser.toPixels(imageTensor, canvas);
    surface.drawArea.appendChild(canvas);

    imageTensor.dispose();
  }
};

function download_csv() { // UTILITY FUNCTION TO DOWNLOAD ALL RESULTS INTO A CSV FILE
  values.unshift(headers);
  let csvContent = "data:text/csv;charset=utf-8," + values.map(e => e.join(",")).join("\n");
  var encodedUri = encodeURI(csvContent);
  var link = document.getElementById("download_table");
  link.setAttribute("href", encodedUri);
  link.setAttribute("download", "tm_results.csv");
  document.body.appendChild(link); 
  link.click(); 
};


// ############################################################
// ############################################################
// DATA COLLECTION FUNCTIONS
// ############################################################
// ############################################################
var counter; // PLACEHOLDER WEBCAM DATASET ITERATION COUNT

function toggle_validation() { // TOGGLES VALIDATION COLLECTION BUTTONS
  var checkBox = document.getElementById("defaultCheck1");
  var val_button1 = document.getElementById("2");
  var val_button2 = document.getElementById("3");

  if (checkBox.checked == true){
    val_button1.className = "m-1 btn btn-primary";
    val_button2.className = "m-1 btn btn-primary";
  } else {
    val_button1.className = "d-none";
    val_button2.className = "d-none";
  }
};

async function getImage() {  // SAVES WEBCAM IMAGE
  const img = await webcam.capture();
  original_images.push(img); 
  const processedImg = tf.tidy(() => img.expandDims(0).toFloat().div(127).sub(1));
  // img.dispose(); // WE DO NOT DISPOSE SINCE WE WANT TO SHOW THEM
  return processedImg;
};

async function add_data(controllerDataset, label) { // ADS SAVED WEBCAM IMAGE TO A GIVEN DATASET
  let img = await getImage();
  controllerDataset.addExample(mobilenet.predict(img), label);
  img.dispose();
};

function collect_images(elem) { // CAPTURES AND SAVES WEBCAM IMAGE CAPTURES - FOR BUTTON
  counter = setInterval(function() {
      switch(elem.id){
        case "0":
          leftSamples++;
          // document.getElementById("leftsamples").innerText = "Looking Left Samples: " + leftSamples;
          document.getElementById("leftsamples").innerText = leftSamples;
          break;
        case "1":
          rightSamples++;
          // document.getElementById("rightsamples").innerText = "Looking Right Samples: " + rightSamples;
          document.getElementById("rightsamples").innerText = rightSamples;
          break;
      }
      label = parseInt(elem.id);
      add_data(dataset, label);
  }, 50);
};


function collect_val_images(elem) { // CAPTURES AND SAVES WEBCAM VALIDATION IMAGE CAPTURES
  counter = setInterval(function() {
      switch(elem.id){
        case "2":
          leftSamplesVal++;
          document.getElementById("leftsamplesVal").innerText = "Looking Left Samples: " + leftSamplesVal;
          break;
        case "3":
          rightSamplesVal++;
          document.getElementById("rightsamplesVal").innerText = "Looking Right Samples: " + rightSamplesVal;
          break;
      }
      label = parseInt(elem.id) - 2;
      add_data(val_dataset, label);
  }, 50);
};

function end() { // MAKES "HOLD AND CAPTURE" POSSIBLE
  clearInterval(counter);
};


// ############################################################
// ############################################################
// LIVE PREDICTION FUNCTIONS
// ############################################################
// ############################################################

async function getPredictedClass() {  // SAVES WEBCAM IMAGE
  let img = await webcam.capture();
  const processedImg = tf.tidy(() => img.expandDims(0).toFloat().div(127).sub(1));
  const activation = mobilenet.predict(processedImg);
  const predictions = model.predict(activation);

  img.dispose()
  processedImg.dispose()
  activation.dispose()

  return [predictions, predictions.as1D().argMax()];
};

async function predict() { // PREDICTS WEBCAM LIVE CAPTURES 
  while (isPredicting) {
    const predictedClass = await getPredictedClass();
    const output_prediction = (await predictedClass[0].data());
    predictProb(output_prediction);

    predictedClass[1].dispose();
    predictedClass[0].dispose();

    await tf.nextFrame();
  }
};

async function startPredicting(){ // STARTS LIVE PREDICTION
	isPredicting = true;
  try {
    webcam = await tf.data.webcam(document.getElementById('predictWebcam'));
  } catch (e) {
    console.log(e);
  }

	predict();
};

function stopPredicting(){ // STOPS LIVE PREDICTION
	isPredicting = false;
	predict();
};

function predictProb(probArray) {
  let left_prob = probArray[0].toFixed(4);
  let right_prob = probArray[1].toFixed(4);

  let left_probbar = Math.round(probArray[0] * 100);
  let right_probbar = Math.round(probArray[1] * 100);

  const left = document.getElementById('leftprob');
  left.innerText = left_prob;

  const right = document.getElementById('rightprob');
  right.innerText = right_prob;

  
  document.getElementById('leftbar').style.width = left_probbar + "%";
  document.getElementById('rightbar').style.width = right_probbar + "%";

  // Implement Color for Bars based on Probabilities
  if (left_probbar > 0.95) {
    document.getElementById("leftbar").className = "progress-bar progress-bar-animated bg-success";
    document.getElementById("rightbar").className = "progress-bar progress-bar-animated";
  } else {
    document.getElementById("leftbar").className = "progress-bar progress-bar-animated";
  }

  if (right_probbar > 0.95) {
    document.getElementById("rightbar").className = "progress-bar progress-bar-animated bg-success";
    document.getElementById("leftbar").className = "progress-bar progress-bar-animated";
  } else {
    document.getElementById("rightbar").className = "progress-bar progress-bar-animated";
  }
};

// ############################################################
// ############################################################
// MAIN INITIALIZATION FUNCTIONS
// ############################################################
// ############################################################
async function init(){// CALLS FOR WEBCAM INSTANCE AND LOADS MOBILENET
  try { 
    webcam = await tf.data.webcam(document.getElementById('webcam'));
  } catch (e) {
    console.log(e);
  }

  mobilenet = await loadMobilenet();
  
  const screenShot = await webcam.capture(); 
  mobilenet.predict(screenShot.expandDims(0)); 
  screenShot.dispose(); 
};

function doTraining(){ // STARTS TRAINING
	train();
};

function start_camera () { // STARTS CAMERA FOR DATA COLLECTION 
  remove_gif();
  init();
};

function remove_gif() { // REMOVES IMAGE WHEN CAMERA IS INITIALIZED 
  var camera_frame = document.getElementById("video_frame");

  camera_frame.className = "row justify-content-center";
};