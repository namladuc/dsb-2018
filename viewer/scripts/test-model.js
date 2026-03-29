const path = require('path');
const ort = require('onnxruntime-node');

async function main() {
  const modelPath = process.argv[2]
    ? path.resolve(process.cwd(), process.argv[2])
    : path.resolve(__dirname, '..', 'public', 'segmentation_model_timm.onnx');

  console.log(`Loading model: ${modelPath}`);

  try {
    const session = await ort.InferenceSession.create(modelPath);
    console.log('Model load: OK');
    console.log('Inputs:', session.inputNames.join(', '));
    console.log('Outputs:', session.outputNames.join(', '));

    const inputName = session.inputNames[0];
    const inputMeta = session.inputMetadata[inputName];
    console.log('Input meta:', JSON.stringify(inputMeta, null, 2));

    await session.release();
    console.log('Session released.');
  } catch (error) {
    console.error('Model load: FAILED');
    console.error(error && error.stack ? error.stack : error);
    process.exitCode = 1;
  }
}

main();
