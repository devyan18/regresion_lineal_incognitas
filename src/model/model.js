import * as tf from '@tensorflow/tfjs'
const result = document.getElementById('result')

const model = () => {
	const x1 = [1,2,3,4,5]
	const x2 = [2,4,6,8,10]

	const x1s = tf.tensor1d(x1)
	const x2s = tf.tensor1d(x2)

	const test = [2,4,6]

	const model = tf.sequential()
	
	model.add(tf.layers.dense({ units: 1, inputShape: [1] }))
	model.compile({ loss: 'meanSquaredError', optimizer: 'sgd', metrics: ['accuracy'] })
	
	model.fit(x1s, x2s, {epochs: 350})
		.then(() => {
			for(let i=0;i<test.length;i++){
				console.log(model.predict(tf.tensor1d([test[i]])))
				result.innerHTML += `<p>[${test[i]}, ${model.predict(tf.tensor1d([test[i]])).dataSync()}]</p>`
			}
		})
}
export default model