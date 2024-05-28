import { Component,OnInit } from '@angular/core';
import { Product } from '../modals/Product';
import { ProductsService } from '../services/products.service';
import {MatSnackBar} from '@angular/material/snack-bar';

@Component({
  selector: 'app-store-page',
  templateUrl: './store-page.component.html',
  styleUrls: ['./store-page.component.scss']
})
export class StorePageComponent implements OnInit{
  categories = [
    {name: 'All', color: 'accent',selected:"true"},
    {name: 'Food', color: 'primary',selected:"false"},
    {name: 'Toys', color: 'warn',selected:"false"},
    {name: 'Medication', color: 'primary',selected:"false"},
  ];

  products:Product[]=[];
  quantities:number[]=[];
  selectedCategory=this.categories[0].name;

  constructor(public productsService:ProductsService,private matSnackBar:MatSnackBar){}

  ngOnInit(){
    this.products=this.productsService.getProducts(this.selectedCategory);
    for(let e of this.products){
      this.quantities.push(1);
    }
  }

  selectCategory(category:any){
    this.selectedCategory=category;
    this.products=this.productsService.getProducts(this.selectedCategory);
  }

  addToBasket(id:number,quantity:number){
    this.productsService.addToBasket(id,quantity)
    this.matSnackBar.open("Added to cart!","Okay!")
  }

}
