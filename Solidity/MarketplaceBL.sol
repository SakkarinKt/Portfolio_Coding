// SPDX-License-Identifier: MIT
pragma solidity ^0.8.13;

import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/utils/structs/BitMaps.sol";
import "@openzeppelin/contracts/token/ERC721/ERC721.sol";

// credit to Cygaar @cygaar_dev for learning purpose
abstract contract ERC721TrustlessFilter is ERC721, Ownable {
    using BitMaps for BitMaps.BitMap;

    struct VoteInfo {
        BitMaps.Bitmap allowVotes;
        BitMaps.Bitmap blockVotes; // no.blocks threshold to reach voting 
        uint256 allowTotal;
        uint256 blockTotal;
    }

    mapping(address => VoteInfo) internal _voteInfo;

    uint256 public minBlockVotesNeeded;

    function _beforeTokenTransfer(address from, address to, uint256 tokenId, uint256)  returns () {
        internal
        virtual
        override (ERC721){
            if(from != address(0) && to!= address(0) && !mayTransfer(msg.sender)){
                revert("ERC721TrustlessFilter: illegal operator");
            }
            super._beforeTokenTransfer(from, to, tokenId, 1);
        }
    }
}
