<template>
  <div class="uploader">
    <uploader :options="options" class="uploader-example">
      <uploader-unsupport></uploader-unsupport>
      <uploader-drop>
        <h2>上传文件</h2>
        <h5>选择或拖动文件（文件夹）到此处</h5>
        <uploader-btn>选择文件</uploader-btn>
        <uploader-btn :attrs="attrs">选择图片</uploader-btn>
        <uploader-btn :directory="true">选择文件夹</uploader-btn>
      </uploader-drop>
      <uploader-list></uploader-list>
    </uploader>

    <div class="result"></div>
  </div>
</template>

<script>
import axios from "axios";
import Swiper from "swiper";

export default {
  data() {
    return {
      options: {
        // https://github.com/simple-uploader/Uploader/tree/develop/samples/Node.js
        target: "//localhost:5000/predict",
        testChunks: false
      },
      attrs: {
        accept: "image/*"
      }
    };
  },

  methods: {
    getPredFromBackend() {}
  }
};
</script>

<style>
.uploader-example {
  width: 880px;
  padding: 15px;
  margin: 40px auto 0;
  font-size: 12px;
  box-shadow: 0 0 10px rgba(0, 0, 0, 0.4);
  background: url(./assets/img/buaa.jpeg) no-repeat;
  background-size: 100% 100%;
}
.uploader-example .uploader-btn {
  margin-right: 4px;
}
.uploader-example .uploader-list {
  max-height: 440px;
  overflow: auto;
  overflow-x: hidden;
  overflow-y: auto;
}
.uploader-example .uploader-drop {
  text-align: center;
  /* width: 540px; */
  margin: auto;
}
.result {
  text-align: center;
}
</style>
