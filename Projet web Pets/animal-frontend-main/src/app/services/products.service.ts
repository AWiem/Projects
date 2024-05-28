import { Injectable } from '@angular/core';
import { Product } from '../modals/Product';
import * as myGlobals from '../globals';

@Injectable({
  providedIn: 'root'
})
export class ProductsService {
  products:Product[]=[
    {id:1,name:"Happy Dog Bites 750g",price:"35.4",category:"Food",description:"Happy Dog Super Bite,4 in 1, Flavor Healthy & Tasty Goody Bag Energy Treat Snack for Dogs [Liver, Lamb, Calcium, Chlorophyll]",img:"../../assets/Happy Dog Bites.jpg"},
    {id:2,name:"Bakers super Food",price:"45.2",category:"Food",description:"energy treats made from vegetable pro-line which is perfect for dogs allergic to meat. it is suitable for all breeds and ages of dogs. made from high quality ingredients its texture can be flexible and fun to chew by your dogs. ",img:"../../assets/Bakers Super Food.jpg"},
    {id:3,name:"Doggy chew Toy",price:"7.6",category:"Toys",description:" a chew toy is the ideal dog toy for destroyers. These re-enforced toys are made to withstand your pup's powerful jaws and sharp teeth. Typically made from rubber, chew toys are often more expensive than other dog toys.",img:"../../assets/Dog toy.jpg"},
    {id:4,name:"Kitten Toy",price:"5.2",category:"Toys",description:"What can a cat love more that a mouse",img:"../../assets/Cat toy.jpg"},
    {id:5,name:"Cold Shield",price:"9.8",category:"Medication",description:"Clean your kitten's nostrils and ease its cold",img:"../../assets/cold Med Cats.jpg"},
    {id:6,name:"Dog Spray",price:"15.2",category:"Medication",description:"Efficient spray for dogs",img:"../../assets/Dog spray.jpg"},
  ];

  constructor() { }

  getProducts(category:string){
    let filteredProducts:Product[]=[];
    if(category=="All")
    {
      filteredProducts=this.products;
    }
    else
    {
      for(let product of this.products)
      {
        if(product.category==category)
        {
          filteredProducts.push(product);
        }
      }
    }
    return filteredProducts;
  }
  getProduct(id:number){
    let returnProduct:any=null;
    for(let product of this.products)
    {
      if(product.id==id)
      {
        returnProduct=product;
      }
    }
    return returnProduct;
  }
  addToBasket(id:number,quantity:number){
    myGlobals.basket.push({id,quantity})
  }
  removeFromBasket(id:number){
    myGlobals.basket.forEach((product,index)=>{
      if(product.id==id) myGlobals.basket.splice(index,1);
   });
  }
}
