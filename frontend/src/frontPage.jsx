import image from "./color_005.jpg"
import crop from "./singleImage.jpg"
import colorCrop from "./singleCropImageColor.jpg"
import rightArrow from "./arrow_right.webp"
import arch from "./neural_network_architecture.jpeg"

function Images(){

    return(
        <>
        
        <div className="inline-flex justify-center relative">
            <img src={image} className="mr-20"/>

            <div className="absolute font-bold">
                <p>Original Image</p>

            </div>
        
        </div>
        
        <img src={rightArrow} className="inline-flex l-20 w-20 mr-20"/>

        <div className="items-center justify-center relative">
            <img src={colorCrop} className="inline-flex mr-20"/>
            <img src={rightArrow} className="inline-flex l-20 w-20 mr-20"/>
            <img src={crop} className="inline-flex mr-20"/>
            <img src={rightArrow} className="inline-flex l-20 w-20 mr-20"/>
            <img src={arch} className="inline-flex l-1/3 w-1/3 mr-20"/>

        </div>
        
        </>

    )
}


export default Images