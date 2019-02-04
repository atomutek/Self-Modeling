from tensorflow_src import main_tf as main
# from pytorch_src import main_pytorch as main

if __name__ == '__main__':
    args = main.parse_args()
    # Run actual script.
    main.run(**args)
