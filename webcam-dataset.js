class WEBCAMDataset {
    // THIS CLASS HANDLES THE COLLECTION AND CREATION OF A DATASET IN A FORMAT
    // APPROPIATE FOR TENSORLFOW.
    constructor() {
      this.labels = [];
    }
  
    addExample(example, label) {
      if (this.xs == null) {
        // For the first example that gets added, keep example and y so that the
        // ControllerDataset owns the memory of the inputs. This makes sure that
        // if addExample() is called in a tf.tidy(), these Tensors will not get
        // disposed.
        this.xs = tf.keep(example);
        this.labels.push(label);
      } else {
        const oldX = this.xs;
        this.xs = tf.keep(oldX.concat(example, 0));
        this.labels.push(label);
        oldX.dispose();
      }
    }
    
    encodeLabels(numClasses) {
      // One-hot encode the label
      for (var i = 0; i < this.labels.length; i++) {
        if (this.ys == null) {
          this.ys = tf.keep(tf.tidy(
              () => {return tf.oneHot(
                  tf.tensor1d([this.labels[i]]).toInt(), numClasses)}));
        } else {
          const y = tf.tidy(
              () => {return tf.oneHot(
                  tf.tensor1d([this.labels[i]]).toInt(), numClasses)});
          const oldY = this.ys;
          this.ys = tf.keep(oldY.concat(y, 0));
          oldY.dispose();
          y.dispose();
        }
      }
    }
  }