// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

const path = require('path');
const CopyPlugin = require('copy-webpack-plugin');

module.exports = () => {
  return {
    target: ['web'],
    mode: 'production',
    entry: path.resolve(__dirname, 'src', 'main.js'),
    output: {
      path: path.resolve(__dirname, 'dist'),
      filename: 'bundle.min.js',
      library: {
        type: 'umd',
      },
      clean: true,
    },
    plugins: [
      new CopyPlugin({
        patterns: [
          { from: 'node_modules/onnxruntime-web/dist/*.wasm', to: '[name][ext]' },
          { from: 'public/segmentation_model_timm.onnx', to: '[name][ext]' },
          { from: 'webpack.index.html', to: 'index.html' },
        ],
      }),
    ],
  };
};
