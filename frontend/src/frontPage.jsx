import image from "./color_005.jpg"
import crop from "./singleImage.jpg"
import colorCrop from "./singleCropImageColor.jpg"

function Images(){

    return(
        <>
        
        <div className="items-center justify-center">
            <img src={image} className="inline-flex mr-20"/>
            <img src={colorCrop} className="inline-flex mr-20"/>
            <img src={crop} className="inline-flex"/>
        
        </div>
        
        </>

    )
}


export default Images