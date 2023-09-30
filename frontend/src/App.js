import logo from './logo.svg';
import './App.css';
import {useState, useEffect} from 'react'
import Images from './frontPage';


function App() {
  const basePath = 'C:/Users/Crolw/OneDrive/Documents/GitHub/lip-reading/MIRACL-VC1_all_in_one/'

  const [image, setImage] = useState(0);

  function pad(num) {
    return (num < 10) ? '0' + num.toString() : num.toString();
}

  const getRandomNumber = (max) => {
    return Math.floor(Math.random() * max);
  }

  const predict = () => {
    console.log(image)
    const types = ['phrases', 'words'];

    var person = getRandomNumber(10);
    var expNum = getRandomNumber(10);
    var repeatNum = getRandomNumber(10);
    var typeIndex = getRandomNumber(2);
    var ranNum = getRandomNumber(8);

    while (person == 3)
      person = getRandomNumber(10);

    const fileName = 'color_0' + `${ranNum}` + '.jpg'

    const path = `F${person}/${types[typeIndex]}/${expNum}/${repeatNum}/color_001.jpg`

    setImage(path);
  }
  
  return (
    <div className="App">
    <h1 className="text-3xl font-bold font-serif">Lip Interpretation Process</h1>
      
      <Images/>
      <div className='image'>
        <img src={image}/>
      </div>

      <button className='btn' onClick={predict}> Predict </button>

    </div>
  );

}

export default App;
