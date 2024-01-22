const CONTROLS = ['left', 'right'];
const CONTROL_CODES = ['left', 'right'];

const lossStatusElement = document.getElementById('loss-status');
const accStatusElement = document.getElementById('accuracy-status');

// Set hyper params from UI values.
const learningRateElement = document.getElementById('learningRate');
const batchSizeFractionElement = document.getElementById('batchSizeFraction');
const epochsElement = document.getElementById('epochs');
const denseUnitsElement = document.getElementById('dense-units');

const statusElement = document.getElementById('status');