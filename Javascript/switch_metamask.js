const { ethereum } = window;

if (typeof ethereum !== 'undefined' && ethereum.isMetaMask){
    console.log("MetaMask is installed")
}
else{
    console.log("MetaMask is not installed")
}

try{
    await ethereum.request({
        method: 'wallet_swithEthereumChain',
        params: [{chainId: '0xf00'}],
    });
}
catch(switchError){
    if(switchError.code === 4902){
        // do something
    }
}
// prompt user to switch to Polygon chain
const SwitchChainToPolygon = () => {
    const { ethereum } = window;
    if(typeof ethereum !== 'undefined' && ethereum.isMetaMask) 
        return;
    try{
        await ethereum.request({
            method: 'wallet_switchEthereumChain',
            params: [{ chainId: '0x89'}],
        });
    }
    catch(switchError){
        if(switchError.code === 4902){
            console.log("Polygon Chain has't been added to the wallet!")
        }
    }
}