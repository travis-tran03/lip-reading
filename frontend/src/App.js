import logo from './logo.svg';
import './App.css';
import {useState, useEffect} from 'react'

function App() {
  const [number, setNumber] = useState(0);
  const [number2, setNumber2] = useState(0);

  const add = () => {
    setNumber(number + 1);
    console.log(number)
  }

  useEffect(function () {
    setNumber2(number2 + 10);
    console.log('running')
  }, [])

  return (
    <div className="App">
      
      <div>Hello</div>
      <div>Number: {number}</div>
      <div>Number 2: {number2}</div>
      <button onClick={() => {add()}}> Button </button>
    </div>
  );
}

export default App;
