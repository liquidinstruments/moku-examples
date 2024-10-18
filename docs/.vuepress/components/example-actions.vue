<template>
    <div class="button-row">
      <a :href="colabUrl" class="nav-link action-button" target="_blank">
        <span class="icon">
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"  height="24" width="24">
                <path fill="currentColor" d="M16.9414 4.9757a7.033 7.033 0 0 0 -4.9308 2.0646 7.033 7.033 0 0 0 -0.1232 9.8068l2.395 -2.395a3.6455 3.6455 0 0 1 5.1497 -5.1478l2.397 -2.3989a7.033 7.033 0 0 0 -4.8877 -1.9297zM7.07 4.9855a7.033 7.033 0 0 0 -4.8878 1.9316l2.3911 2.3911a3.6434 3.6434 0 0 1 5.0227 0.1271l1.7341 -2.9737 -0.0997 -0.0802A7.033 7.033 0 0 0 7.07 4.9855zm15.0093 2.1721 -2.3892 2.3911a3.6455 3.6455 0 0 1 -5.1497 5.1497l-2.4067 2.4068a7.0362 7.0362 0 0 0 9.9456 -9.9476zM1.932 7.1674a7.033 7.033 0 0 0 -0.002 9.6816l2.397 -2.397a3.6434 3.6434 0 0 1 -0.004 -4.8916zm7.664 7.4235c-1.38 1.3816 -3.5863 1.411 -5.0168 0.1134l-2.397 2.395c2.4693 2.3328 6.263 2.5753 9.0072 0.5455l0.1368 -0.1115z"/>
            </svg>
        </span>
        Open in Google Colab
      </a>
  
      <a :href="githubUrl" class="nav-link action-button" target="_blank">
        <span class="icon">
          <svg xmlns="http://www.w3.org/2000/svg" height="24" width="24" viewBox="0 0 24 24">
            <path fill="currentColor" d="M12 0C5.37 0 0 5.37 0 12c0 5.3 3.438 9.799 8.207 11.387.6.111.793-.261.793-.577v-2.165c-3.337.725-4.033-1.434-4.033-1.434-.546-1.388-1.334-1.758-1.334-1.758-1.09-.745.082-.729.082-.729 1.204.083 1.838 1.243 1.838 1.243 1.07 1.834 2.808 1.305 3.492.998.109-.774.419-1.305.762-1.605-2.665-.303-5.466-1.332-5.466-5.926 0-1.309.467-2.381 1.235-3.222-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.3 1.232a11.487 11.487 0 0 1 6 0c2.29-1.554 3.297-1.232 3.297-1.232.654 1.652.243 2.873.119 3.176.771.841 1.235 1.914 1.235 3.222 0 4.608-2.805 5.618-5.479 5.916.43.373.816 1.106.816 2.229v3.306c0 .32.192.694.8.576C20.563 21.799 24 17.3 24 12 24 5.37 18.63 0 12 0Z"/>
          </svg>
        </span>
        Open in GitHub
      </a>
  
      <a :href="downloadUrl" class="nav-link action-button" target="_blank" download="filename">
        <span class="icon">
          <svg xmlns="http://www.w3.org/2000/svg" height="24" width="24" viewBox="0 0 24 24">
            <path fill="currentColor" d="M12 16.5l6-6H15V3H9v7.5H6l6 6ZM6 21v-2.5h12V21H6Z"/>
          </svg>
        </span>
        Download
      </a>
    </div>
  </template>
  
  <script>
  export default {
    name: "example-actions",
    props: {
      directory: {
        type: String,
        required: true,
      },
      filename: {
        type: String,
        required: true,
      }
    },
    computed: {
    colabUrl() {
      return `https://colab.research.google.com/github/liquidinstruments/moku-examples/blob/main/${this.directory}/${this.filename}`;
    },
    githubUrl() {
      return `https://github.com/liquidinstruments/moku-examples/blob/main/${this.directory}/${this.filename}`;
    },
    downloadUrl() {
      return `https://github.com/liquidinstruments/moku-examples/raw/main/${this.directory}/${this.filename}`;
    },
  },
}
  </script>
  
  <style scoped>
  .button-row {
    display: flex;
    gap: 15px;
    justify-content: flex-start;
    align-items: center;
    margin: 2rem 0;
  }
  
  .nav-link {
    text-decoration: none;
    color: #333;
    font-weight: bold;
    display: flex;
    align-items: center;
    padding: 5px 10px;
    border-radius: 4px;
    transition: background-color 0.2s ease;
    font-size: 12px;
  }
  
  .nav-link:hover {
    background-color: #f0f0f0;
  }
  
  .action-button {
    background-color: #0073e6;
    color: white;
    border-radius: 0px;
  }
  
  .action-button:hover {
    background-color: #005bb5;
  }
  
  .icon {
    display: inline-block;
    margin-right: 10px;
  }
  
  .icon svg {
    width: 20px;
    height: 20px;
    fill: currentColor;
  }
  </style>
  