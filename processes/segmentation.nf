process predict {
    label 'withGPU'

    container { params.segmentation_container }
    cpus { params.segmentation_cpus }
    memory { params.segmentation_memory }

    input:
    val(image_path)
    val(ch)
    val(scale)
    val(model_path)
    val(output_path)
    val(min_segsize)
    val(diameter)

    output:
    tuple val(image_path), val(output_path)
    
    script:
    ofile = new File(output_path)
    parent_dir = ofile.getParent()
    if (!new File(parent_dir).exists()) {
        new File(parent_dir).mkdirs()
    }
    """
    /entrypoint.sh segmentation -i $image_path -o $output_path -n $ch/$scale --min $min_segsize --diameter $diameter --model $model_path
    """
}