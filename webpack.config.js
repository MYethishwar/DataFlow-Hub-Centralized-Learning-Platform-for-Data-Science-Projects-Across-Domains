const path = require('path');

module.exports = {
    mode: 'development',  // Change to 'production' for production build
    entry: './static/js/app.js',
    output: {
        filename: 'bundle.js',
        path: path.resolve(__dirname, 'static/js/dist'),
    },
    devtool: 'source-map',  // Helps with debugging
    module: {
        rules: [
            {
                test: /\.(js|jsx)$/,
                exclude: /node_modules/,
                use: {
                    loader: 'babel-loader',
                    options: {
                        presets: ['@babel/preset-react']
                    }
                }
            }
        ]
    },
    resolve: {
        extensions: ['.js', '.jsx']
    }
};